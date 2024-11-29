class Node:
    def __init__(self, id, name):
        self.is_root = False
        self.id = id
        self.name = name
        self.parents = []
        self.next = {}  # 使用字典来存储子节点及其关系描述

def build_tree(ns, rs):
    nodes = {node['id']: Node(node['id'], node['name']) for node in ns}

    # 创建邻接表以便于查找关系
    from collections import defaultdict
    adjacency_list = defaultdict(list)
    for relation in rs:
        adjacency_list[relation['start_id']].append((relation['description'], relation['end_id']))

    # 检测循环依赖的DFS函数
    def dfs(node, visited, rec_stack):
        if node.id == 146:
            print(node.id)
        if node.id in visited:
            return False
        if node.id in rec_stack:
            return False
            # raise ValueError(f"Circular dependency detected involving node {node.id}")
        rec_stack.add(node.id)
        for description, neighbor_id in adjacency_list[node.id]:
            neighbor = nodes.get(neighbor_id)
            if not neighbor:
                continue
            if neighbor.id in rec_stack:
                neighbor.is_root = True
            # 如果当前节点不是邻居的父节点，则将其添加为父节点，并将当前节点作为邻居的子节点
            if node not in neighbor.parents:
                neighbor.parents.append(node)
                node.next[neighbor] = description
                dfs(neighbor, visited, rec_stack)
        rec_stack.remove(node.id)
        visited.add(node.id)

    # 开始DFS
    visited = set()
    rec_stack = set()
    for node in nodes.values():
        if not node.parents:  # 从根节点开始
            dfs(node, visited, rec_stack)

    roots = [node for node in nodes.values() if not node.parents or node.is_root]
    return roots

def get_paths(roots, max_depth=5):
    paths = []

    def traverse(node, current_path, visited, depth):
        # 如果递归深度超过最大值，退出递归
        if depth > max_depth:
            return

        # 如果节点已经访问过，说明存在循环依赖，退出递归
        if node in visited:
            return

        # 标记节点为已访问
        visited.add(node)

        # 添加当前节点到路径
        if node.name:
            current_path.append(node.name)

        # 如果没有子节点，说明是叶子节点，把当前路径加入到结果中
        if not node.next:
            path_str = "，".join(current_path)
            paths.append(path_str)
        else:
            # 遍历子节点
            for child, description in node.next.items():
                # 复制当前路径并添加描述和子节点
                new_path = current_path + [description]
                traverse(child, new_path, visited, depth + 1)

        # 递归结束后，移除节点的访问标记
        visited.remove(node)
        if current_path:
            current_path.pop()

    for root in roots:
        traverse(root, [], set(), 0)

    return paths