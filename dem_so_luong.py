import os

data_dir = "./input"  # Thay thế bằng đường dẫn thực tế đến dataset của bạn
categories = [
    "Canker",
    "Dot",
    "Healthy",
    "Mummification",
    "Phytopthora",
    "Rust",
    "Scab",
    "Styler_Root",
]
sets = ["train", "test", "val"]

# Đếm số lượng hình ảnh trong mỗi thư mục
for set_name in sets:
    print(f"--- {set_name.upper()} SET ---")
    for category in categories:
        path = os.path.join(data_dir, set_name, category)
        count = len(os.listdir(path))
        print(f"{category}: {count} images")
    print()
