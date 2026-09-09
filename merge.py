import os
import glob
from datetime import datetime

DATE_PATTERN = r'^([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?\.md$'

def parse_filename_as_date(filename):
    """
    尝试把文件名解析成 datetime
    成功返回 (datetime, 原始文件名)
    失败返回 None
    """
    import re
    match = re.match(DATE_PATTERN, filename.strip())
    if not match:
        return None

    month_str, day_str = match.groups()
    month_str = month_str.capitalize()

    # 英文月份映射
    month_map = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'Aug': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }

    if month_str not in month_map:
        return None

    try:
        dt = datetime(2026, month_map[month_str], int(day_str))
        return dt, filename
    except ValueError:
        # 非法日期（如 April 31st）
        return None


def merge_md_files_by_real_date(
    source_dir='.',
    output_file='APRIL-MAY_工作日记合集.md'
):
    md_files = glob.glob(os.path.join(source_dir, '*.md'))

    parsed_files = []

    for path in md_files:
        filename = os.path.basename(path)
        result = parse_filename_as_date(filename)
        if result:
            parsed_files.append((result[0], path))

    if not parsed_files:
        print("❌ 没有找到符合日期格式的 .md 文件")
        return

    # 按真实日期排序
    parsed_files.sort(key=lambda x: x[0])

    merged_content = []
    merged_content.append("# 工作日记合集（仅日期格式）\n\n")

    for dt, path in parsed_files:
        filename = os.path.basename(path)
        print(f"✅ 合并: {filename}")

        with open(path, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        merged_content.append(f"\n{'='*80}\n")
        merged_content.append(f"# {dt.strftime('%B %d').replace('0', '')}{get_day_suffix(dt.day)}\n")
        merged_content.append(f"{'-'*40}\n\n")
        merged_content.append(content)
        merged_content.append("\n")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("".join(merged_content))

    print(f"\n🎉 合并完成！共 {len(parsed_files)} 个文件")
    print(f"📄 输出文件: {output_file}")


def get_day_suffix(day):
    if 11 <= day <= 13:
        return 'th'
    return {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')


if __name__ == "__main__":
    merge_md_files_by_real_date(
        source_dir='.',
        output_file='APRIL-MAY_工作日记合集.md'
    )