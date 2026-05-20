import json5

from phar_tool_limits import DATA_FOLDER_MAX_SLICE, DATA_FOLDER_MAX_TOTAL_READ
from qwen_agent.tools.base import BaseTool, register_tool

from .folder_name_resolve import resolve_folder_name


@register_tool('data_folder')
class DataFolder(BaseTool):
    description = (
        '按索引读取数据文件夹中的帖子。单次最多读取 '
        f'{DATA_FOLDER_MAX_SLICE} 条；同一任务内累计不超过 {DATA_FOLDER_MAX_TOTAL_READ} 条。'
        '优先读取 topic_clustering 生成的 folder_name_cluster；勿对原始窗大批量顺序扫描。'
    )
    parameters = [{
        'name': 'folder_name',
        'type': 'string',
        'description': '文件夹名称（聚类结果请用 folder_name + "_cluster"）',
        'required': True
    },
                  {
        'name': 'start_idx',
        'type': 'int',
        'description': f'开始索引（包含），单次读取 end_idx - start_idx 不得超过 {DATA_FOLDER_MAX_SLICE}',
        'required': True
    },
                  {
        'name': 'end_idx',
        'type': 'int',
        'description': (
            f'结束索引（不包含）。读原始帖子列表时为帖序号；读 *_cluster 时为簇编号。'
            f'单次跨度不超过 {DATA_FOLDER_MAX_SLICE}'
        ),
        'required': True
    },]
    data_folders = {}
    show_funcs = {}
    read_record = {}
    read_count = 0
    empty_folders: set[str] = set()

    EMPTY_FOLDER_REPLY = (
        "该文件夹内帖子数为 0（post_search 未命中任何帖子）。"
        "请勿再调用 data_folder、topic_clustering 或 post_retrieve；"
        "请调整 keywords/时间窗后重新 post_search，或直接输出空结果 JSON。"
    )

    def initialize(self):
        self.read_count = 0
        self.read_record = {}
        self.data_folders = {}
        self.show_funcs = {}
        self.empty_folders = set()

    def reset_read_budget(self):
        """Clear per-query read counters; keep loaded folders (for API retries)."""
        self.read_count = 0
        self.read_record = {}

    def mark_folder_empty(self, folder_name: str) -> None:
        self.empty_folders.add(folder_name)

    def clear_folder_empty(self, folder_name: str) -> None:
        self.empty_folders.discard(folder_name)

    def folder_has_no_posts(self, folder_name: str) -> bool:
        if folder_name in self.empty_folders:
            return True
        posts = self.data_folders.get(folder_name)
        if posts is None:
            return False
        return len(posts) == 0

    def call(self, params: str, **kwargs):
        params = json5.loads(params)
        folder_name, start_idx, end_idx = params['folder_name'], params['start_idx'], params['end_idx']
        start_idx, end_idx = int(start_idx), int(end_idx)
        folder_name = resolve_folder_name(self, folder_name)

        if folder_name not in self.data_folders:
            raise ValueError(f"文件夹'{folder_name}'不存在。请先创建文件夹或检查名称是否正确。")

        if self.folder_has_no_posts(folder_name):
            return self.EMPTY_FOLDER_REPLY

        if start_idx < 0:
            raise ValueError("起始索引不能小于0。")
        if end_idx <= start_idx:
            raise ValueError("结束索引必须大于起始索引。")

        slice_len = end_idx - start_idx
        if slice_len > DATA_FOLDER_MAX_SLICE:
            return (
                f"单次最多读取 {DATA_FOLDER_MAX_SLICE} 条帖子（当前请求 {slice_len} 条）。"
                f"请缩小范围，例如 start_idx={start_idx}, end_idx={start_idx + DATA_FOLDER_MAX_SLICE}；"
                "或改用 post_retrieve（topk≤8）按语义取代表性帖子。"
            )

        n_posts = len(self.data_folders[folder_name])
        if end_idx > n_posts:
            end_idx = n_posts
            slice_len = end_idx - start_idx
            if slice_len <= 0:
                return f"索引超出范围：文件夹共 {n_posts} 条，start_idx={start_idx} 无效。"

        if folder_name not in self.read_record:
            self.read_record[folder_name] = []

        valid_ranges = [[start_idx, end_idx]]
        for si, ei in self.read_record[folder_name]:
            edited = True
            while edited:
                edited = False
                for j, (vsi, vei) in enumerate(valid_ranges):
                    if vsi >= si and ei >= vei:
                        edited = True
                        valid_ranges[j] = [-1, -1]
                        break
                    elif vsi <= si and ei <= vei:
                        edited = True
                        valid_ranges[j] = [-1, -1]
                        valid_ranges.extend([[vsi, si], [ei, vei]])
                        break
                    elif vsi <= si and ei >= vei and si < vei:
                        edited = True
                        valid_ranges[j] = [vsi, si]
                        break
                    elif vsi >= si and ei <= vei and vsi < ei:
                        edited = True
                        valid_ranges[j] = [ei, vei]
                        break
                while [-1, -1] in valid_ranges:
                    valid_ranges.remove([-1, -1])

        if len(valid_ranges) == 0:
            return "***该数据已读取过，请查看历史输出***"

        new_reads = sum(int(vei) - int(vsi) for vsi, vei in valid_ranges)
        if self.read_count + new_reads > DATA_FOLDER_MAX_TOTAL_READ:
            remaining = max(0, DATA_FOLDER_MAX_TOTAL_READ - self.read_count)
            return (
                f"本任务已累计读取 {self.read_count} 条，再读 {new_reads} 条将超过上限 "
                f"{DATA_FOLDER_MAX_TOTAL_READ}（尚可读取约 {remaining} 条）。"
                "请基于已有工具输出归纳，或使用 post_retrieve（topk≤8）补充少量代表性帖子。"
            )

        cluster_hint = ""
        if folder_name.endswith("_cluster"):
            n_clusters = len(self.data_folders[folder_name])
            cluster_hint = (
                f"【聚类文件夹】共 {n_clusters} 个簇；start_idx/end_idx 为簇编号（半开区间），"
                f"不是帖子的顺序号。\n\n"
            )

        result = cluster_hint
        for vsi, vei in valid_ranges:
            result += self.show_funcs[folder_name](self.data_folders[folder_name], vsi, vei) + '\n'
            self.read_count += int(vei) - int(vsi)
            self.read_record[folder_name].append([vsi, vei])

        if self.read_count >= DATA_FOLDER_MAX_TOTAL_READ:
            result += (
                f"\n\n***已达本任务 data_folder 读取上限（{DATA_FOLDER_MAX_TOTAL_READ} 条），"
                "请勿继续调用 data_folder，请直接输出 JSON。***"
            )

        return result
