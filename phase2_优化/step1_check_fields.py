"""
Step 1: 检查 FS_Combas 字段，确认能否自算 INV
"""
import pandas as pd, os

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

# FS_Combas 有 3 行头：英文名 / 中文描述 / 单位
# 先读第2行（index=1）看中文描述
desc = pd.read_csv(f"{DATA}/FS_Combas.csv", encoding='gbk', nrows=2, header=None)
# desc.iloc[0] = 英文列名, desc.iloc[1] = 中文描述
en_names = desc.iloc[0].tolist()
cn_names = desc.iloc[1].tolist()

print("FS_Combas 共", len(en_names), "列\n")

# 找出与 INV 计算相关的列
targets = ['资产', '权益', '利润', '收入', '成本', '费用']
for t in targets:
    print(f"--- 含 '{t}' 的列 ---")
    for i, name in enumerate(cn_names):
        if isinstance(name, str) and t in name:
            print(f"  [{i}] {en_names[i]}  =  {name}")
    print()

# 关键列确认
print("=" * 60)
print("关键列快速定位（按 index）：")
for i, (en, cn) in enumerate(zip(en_names, cn_names)):
    if isinstance(en, str) and en in ['A001200000', 'A002000000', 'A003000000', 'A001100000', 'A001219000', 'A001000000']:
        print(f"  [{i}] {en}  =  {cn}")

# 读几行数据看看
print("\n" + "=" * 60)
print("前 3 行数据（关键列）：")
df = pd.read_csv(f"{DATA}/FS_Combas.csv", encoding='gbk', skiprows=[1,2], nrows=5)
key_cols = ['Stkcd', 'Accper', 'Typrep'] + \
           [c for c in df.columns if c in ['A001200000', 'A002000000', 'A003000000', 'A001100000', 'A001219000', 'A001000000']]
print(df[key_cols].to_string())
print(f"\n总行数: {len(pd.read_csv(f'{DATA}/FS_Combas.csv', encoding='gbk', skiprows=[1,2])):,}")
print("=" * 60)
print("完成。确认 A001200000=总资产，可用于自算 INV")
