from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "train_milk10k_transformer.py"


def test_training_script_exists_and_has_entrypoint():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "class MilkTriFormer" in text
    assert "def main()" in text
    assert 'if __name__ == "__main__"' in text


def test_milk10k_class_contract():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    class_names = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "CLASS_NAMES":
                    class_names = ast.literal_eval(node.value)
    assert class_names == ["AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF", "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"]


def test_repository_folders_exist():
    for rel in ["configs", "data", "docs", "scripts", "templates", "checkpoints", "logs", "outputs"]:
        assert (ROOT / rel / "README.md").exists() or (ROOT / rel).exists()
