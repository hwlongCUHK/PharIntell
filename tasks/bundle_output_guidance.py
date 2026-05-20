"""Shared agent/bundle prompt fragments for CCBI & DTS bundle tasks."""

# Agent mode: tools must include clustering before final JSON.
CLUSTER_WORKFLOW_AGENT = """\
工具顺序（强制，不得跳步或仅凭 post_retrieve 直接交卷）：
1. post_search
2. topic_clustering（参数 folder_name 与 post_search 返回完全一致）
3. data_folder 读取聚类结果文件夹（folder_name + "_cluster"）；start_idx/end_idx 为**簇编号**（半开区间），不是帖序号
4. 可选 post_retrieve：仅作语义补充，**不能替代**步骤 2–3

归纳时：优先**一个输出条目对应一个话题簇**（可合并语义几乎相同的相邻簇）；evidence 须来自该簇内帖子原文，每条≤2，且尽量不同帖。"""

VARIABLE_COUNT_OUTPUT = """\
输出条数（不要固定 5 条）：
- 条数 N = 你有把握区分、且证据充分的主题/簇数量（1≤N≤min(聚类簇数 K, 8)）；**禁止为凑数硬拆**。
- 若帖子高度同质（如全是同一套症状清单模板），可只输出 1–2 条，甚至 1 条。
- post_search 为 0 条时，输出空数组 []。
- 合并语义极近的簇后再输出；相邻条目的短标签须明显不同；rank 从 1 连续编号到 N。"""

# Bundle eval mode: posts/clusters pre-supplied in user message.
VARIABLE_COUNT_BUNDLE = """\
输出条数（不要固定 5 条）：
- 根据用户消息中帖子/主题的**异质化程度**决定条数 N；有明显区分才各占一条。
- 高度同质时 N 可为 1–2；完全无有效内容时 N=0（空数组）。
- 禁止重复或仅换词顺序的条目；rank 从 1 连续编号到 N；每条 evidence≤2 且须来自给定帖子。"""
