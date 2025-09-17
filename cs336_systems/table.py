import pandas as pd

def generate_md_table(data, columns):
    """
    根据给定的数据和列名生成一个 md 表格

    Args: 
        data: 一个列表的列表，每个子列表都代表了一行数据
        columns: 一个包含列名的列表

    Returns:
        一个字符串，代表了 md 格式的表格
    """
    df = pd.DataFrame(data, columns=columns)
    md_table = df.to_markdown(index=False)
    return md_table

# example_data = [
#     ['small', 768, 3072, 12, 12],
#     ['medium', 1024, 4096, 24, 16]
# ]

# example_columns = ["Size", "d_model", "d_ff", "num_layers", "num_heads"]

# my_table = generate_md_table(example_data, example_columns)

# print(my_table)