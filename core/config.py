from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]   # repo 根目录
DATA_DIR = ROOT_DIR / "data"
POLICY_DIR = ROOT_DIR / "policies"
MODEL_DIR = ROOT_DIR / "models"

def resolve_demo_csv(user_path: str) -> str:
    """
    兼容：
    - 绝对路径
    - 相对 repo 根目录（policy_eval_data.csv / data/train.csv）
    - 仅文件名（train.csv / policy_eval_data.csv）：先根目录，再 data/
    """
    p = Path(str(user_path).strip())

    if not p.name:
        return str((ROOT_DIR / "policy_eval_data.csv").resolve())

    # 1) 绝对路径
    if p.is_absolute() and p.exists():
        return str(p)

    # 2) 相对 repo 根目录
    cand = (ROOT_DIR / p).resolve()
    if cand.exists():
        return str(cand)

    # 3) 相对 data/
    cand = (DATA_DIR / p.name).resolve()
    if cand.exists():
        return str(cand)

    # 兜底：返回根目录拼接（让报错更直观）
    return str((ROOT_DIR / p).resolve())
