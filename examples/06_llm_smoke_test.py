from app.agent.llm import get_llm


def main() -> None:
    """
    调用一次真实模型，验证第十章 LLM 链路。
    """

    model = get_llm()

    print("=" * 70)
    print("Globex LLM 连通性测试")
    print("=" * 70)

    response = model.invoke(
        "请只回复：Globex LLM 配置成功"
    )

    print("模型返回：")
    print(response.content)

    print("=" * 70)


if __name__ == "__main__":
    main()