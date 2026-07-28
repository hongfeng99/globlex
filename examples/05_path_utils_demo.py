from app.utils.path_utils import (
    create_session_dir,
    create_uploaded_dir,
    get_project_root,
    resolve_output_path,
    resolve_uploaded_path,
)
from app.utils.thread_ctx import (
    bind_thread_context,
)


def main() -> None:
    """
    演示 Globex 的路径管理流程。
    """

    thread_id = "thread-demo"

    session_dir = create_session_dir(
        thread_id
    )

    uploaded_dir = create_uploaded_dir(
        thread_id
    )

    print("=" * 70)
    print("Globex 路径工具演示")
    print("=" * 70)

    print(f"项目根目录：{get_project_root()}")
    print(f"会话输出目录：{session_dir}")
    print(f"上传目录：{uploaded_dir}")

    with bind_thread_context(
        thread_id=thread_id,
        session_dir=session_dir,
    ):
        report_file = resolve_output_path(
            "reports/final.md",
            create_parent=True,
        )

        report_file.write_text(
            "# Globex 演示报告\n\n"
            "路径工具已经正常工作。\n",
            encoding="utf-8",
        )

        print(f"已生成报告：{report_file}")

    uploaded_file = resolve_uploaded_path(
        thread_id=thread_id,
        relative_path="documents/demo.txt",
        create_parent=True,
    )

    uploaded_file.write_text(
        "这是模拟的上传文件。",
        encoding="utf-8",
    )

    print(f"已生成上传文件：{uploaded_file}")

    print("=" * 70)


if __name__ == "__main__":
    main()