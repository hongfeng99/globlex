from pathlib import Path

from app import __version__


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_package_can_be_imported() -> None:
    """
    验证app包能够被正常导入。
    """

    assert __version__ == "0.1.0"


def test_required_directories_exist() -> None:
    """
    验证基础项目目录已经创建。
    """

    required_directories = [
        "app/agent",
        "app/api",
        "app/tools",
        "app/recall",
        "app/memory",
        "app/compress",
        "app/eval",
        "app/prompt",
        "app/utils",
        "frontend",
        "docker",
        "examples",
        "tests",
        "data",
        "output",
        "uploaded",
    ]

    for relative_path in required_directories:
        directory = PROJECT_ROOT / relative_path

        assert directory.is_dir(), (
            f"缺少项目目录：{relative_path}"
        )


def test_required_configuration_files_exist() -> None:
    """
    验证项目基础配置文件已经创建。
    """

    required_files = [
        "pyproject.toml",
        ".gitignore",
        ".env.example",
    ]

    for relative_path in required_files:
        file_path = PROJECT_ROOT / relative_path

        assert file_path.is_file(), (
            f"缺少配置文件：{relative_path}"
        )