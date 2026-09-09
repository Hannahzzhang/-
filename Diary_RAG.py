#!/usr/bin/env python3
"""
Diary BM25 RAG
一个零 Token、纯本地的中文日记检索系统。
"""

import os
import re
import pickle
import glob
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass

import jieba
from rank_bm25 import BM25Okapi


@dataclass
class SearchResult:
    filename: str
    text: str
    score: float
    chunk_index: int


class DiaryRAG:
    # 别名 -> 标准词
    ALIAS_MAP = {
        # 合伙人
        "管总": "管总",
        "管老师": "管总",
        "春总": "樱桃",
        "Cherry": "樱桃",
        "cherry": "樱桃",
        # 经理
        "Minnow": "米诺",
        "minnow": "米诺",
        "Mino": "米诺",
        "mino": "米诺",
        # 同事（本身已经是标准名，但为了大小写统一也放进来）
        "Christine": "Christine",
        "christine": "Christine",
        "子月": "子玥",
        "紫月": "子玥",
        "自己": "子玥",
        "Raymond": "Raymond",
        "raymond": "Raymond",
        "Mandi": "Mandi",
        "mandi": "Mandi",
        "Mandy": "Mandi",
    }
    
    STOPWORDS = set([
        "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也",
        "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这", "那",
        "啊", "呢", "吧", "吗", "哦", "嗯", "哎", "哎呀", "然后", "就是", "那个", "这个", "什么",
        "时候", "今天", "现在", "觉得", "感觉", "其实", "可能", "应该", "真的", "非常", "比较",
        "还有", "而且", "但是", "不过", "因为", "所以", "虽然", "但是", "如果", "只是", "一下",
        "比如", "像是", "那种", "那种", "之类", "等等",
        # 英文停用词
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "need", "dare",
        "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
        "from", "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "under", "again", "further", "then", "once", "here",
        "there", "when", "where", "why", "how", "all", "each", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not", "only", "own",
        "same", "so", "than", "too", "very", "just", "and", "but", "if", "or",
    ])
    
    def __init__(self, diary_dir: str, cache_path: str = ".diary_bm25_cache.pkl"):
        self.diary_dir = Path(diary_dir)
        self.cache_path = Path(cache_path)
        self.chunks: List[Tuple[str, str]] = []  # [(filename, raw_text), ...]
        self.tokenized: List[List[str]] = []  # [[token, ...], ...]
        self.bm25 = None
        self.file_mtimes: Dict[str, float] = {}
        
    def normalize_text(self, text: str) -> str:
        """将别名统一替换成标准词，便于索引和搜索对齐"""
        # 按长度降序，避免短词先替换导致长词无法匹配
        for alias, std in sorted(self.ALIAS_MAP.items(), key=lambda x: -len(x[0])):
            text = text.replace(alias, std)
        return text
    
    def split_into_chunks(self, text: str, filename: str) -> List[str]:
        """
        智能切分文本：
        1. 先按段落（\n\n+）切分
        2. 段落过长（>300字）按句子切分
        3. 句子还过长（>200字）强制切分
        """
        if not text.strip():
            return []
            
        # 步骤1：按段落切分
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text.strip()) if p.strip()]
        
        chunks = []
        for para in paragraphs:
            if len(para) <= 300:
                chunks.append(para)
                continue
                
            # 步骤2：段落过长，按句子切分
            # 匹配中文句号、问号、感叹号、英文句点（但避免缩写如 Mr.）
            sentences = re.split(r'([。！？\n]+|(?<![A-Za-z])\.(?!\d))', para)
            # 把分隔符拼回去
            sents = []
            i = 0
            while i < len(sentences):
                if i + 1 < len(sentences) and re.match(r'^[。！？\n]+|(?<![A-Za-z])\.(?!\d)$', sentences[i+1]):
                    sents.append(sentences[i] + sentences[i+1])
                    i += 2
                else:
                    sents.append(sentences[i])
                    i += 1
            
            current = ""
            for s in sents:
                s = s.strip()
                if not s:
                    continue
                if len(current) + len(s) < 200:
                    current += s
                else:
                    if current:
                        chunks.append(current)
                    # 如果单句还超过200字，强制切分
                    if len(s) > 200:
                        for i in range(0, len(s), 200):
                            chunks.append(s[i:i+200])
                        current = ""
                    else:
                        current = s
            if current:
                chunks.append(current)
        
        return [c.strip() for c in chunks if c.strip()]
    
    def tokenize(self, text: str) -> List[str]:
        """分词 + 停用词过滤"""
        # 先归一化别名
        text = self.normalize_text(text)
        # jieba 分词
        tokens = jieba.lcut(text)
        # 清洗
        cleaned = []
        for t in tokens:
            t = t.strip().lower()
            # 过滤纯数字、纯标点、单字（可配置是否保留单字）
            if len(t) == 0:
                continue
            if t in self.STOPWORDS:
                continue
            if re.match(r'^\d+$', t):
                continue
            if re.match(r'^[^\w\s\u4e00-\u9fff]+$', t):  # 纯标点
                continue
            cleaned.append(t)
        return cleaned
    
    def load_files(self) -> int:
        """加载日记文件，返回 chunk 数量"""
        md_files = sorted(self.diary_dir.glob("*.md"))
        self.chunks = []
        self.file_mtimes = {}
        
        for f in md_files:
            try:
                text = f.read_text(encoding='utf-8')
                self.file_mtimes[str(f)] = f.stat().st_mtime
                file_chunks = self.split_into_chunks(text, f.name)
                for chunk in file_chunks:
                    self.chunks.append((f.name, chunk))
            except Exception as e:
                print(f"[警告] 读取文件失败 {f.name}: {e}")
        
        return len(self.chunks)
    
    def build_index(self, force=False):
        """构建 BM25 索引"""
        cache_valid = False
        if not force and self.cache_path.exists():
            try:
                with open(self.cache_path, 'rb') as f:
                    cache = pickle.load(f)
                # 检查文件是否修改过
                if cache.get('mtimes') == self.file_mtimes:
                    self.chunks = cache['chunks']
                    self.tokenized = cache['tokenized']
                    self.bm25 = cache['bm25']
                    cache_valid = True
                    print(f"[缓存加载] 共 {len(self.chunks)} 个片段")
            except Exception:
                pass
        
        if not cache_valid:
            print(f"[构建索引] 正在分词 {len(self.chunks)} 个片段...")
            self.tokenized = [self.tokenize(chunk) for _, chunk in self.chunks]
            self.bm25 = BM25Okapi(self.tokenized)
            
            # 保存缓存
            with open(self.cache_path, 'wb') as f:
                pickle.dump({
                    'chunks': self.chunks,
                    'tokenized': self.tokenized,
                    'bm25': self.bm25,
                    'mtimes': self.file_mtimes,
                }, f)
            print(f"[索引完成] 已保存缓存到 {self.cache_path}")
    
    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """搜索"""
        if self.bm25 is None:
            raise RuntimeError("索引未构建，请先调用 build_index()")
        
        q_tokens = self.tokenize(query)
        if not q_tokens:
            return []
        
        scores = self.bm25.get_scores(q_tokens)
        # 取 top_k
        top_indices = scores.argsort()[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            filename, text = self.chunks[idx]
            results.append(SearchResult(
                filename=filename,
                text=text,
                score=float(scores[idx]),
                chunk_index=int(idx)
            ))
        return results
    
    def format_results(self, results: List[SearchResult]) -> str:
        """格式化输出结果"""
        if not results:
            return "未找到相关内容。"
        
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"--- 结果 {i} | {r.filename} | 相关度: {r.score:.2f} ---")
            # 控制输出长度，避免太长
            text = r.text.replace('\n', ' ')
            if len(text) > 500:
                text = text[:500] + "..."
            lines.append(text)
            lines.append("")
        return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="日记 BM25 RAG")
    parser.add_argument("--dir", default=".", help="日记文件夹路径")
    parser.add_argument("--topk", type=int, default=5, help="返回结果数量")
    parser.add_argument("--rebuild", action="store_true", help="强制重建索引")
    parser.add_argument("--query", "-q", default=None, help="直接查询，不传则进入交互模式")
    args = parser.parse_args()

    rag = DiaryRAG(args.dir)
    n_chunks = rag.load_files()
    if n_chunks == 0:
        print("[错误] 未找到任何 .md 日记文件。")
        return
    rag.build_index(force=args.rebuild)

    if args.query:
        results = rag.search(args.query, top_k=args.topk)
        print(rag.format_results(results))
        return

    # 交互模式
    print("\n===== 日记 BM25 RAG 已就绪 =====")
    print("提示: 输入 query 搜索 | 输入 'quit' 退出 | 输入 'rebuild' 重建索引\n")
    while True:
        try:
            query = input("🔍 Query > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("再见!")
            break
        if query.lower() == "rebuild":
            rag.build_index(force=True)
            continue

        results = rag.search(query, top_k=args.topk)
        print(rag.format_results(results))
        print()


if __name__ == "__main__":
    main()