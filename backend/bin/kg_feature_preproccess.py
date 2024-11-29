import click

from service.kg_llm_service import KgLmmService

kg_llm_service = KgLmmService()

@click.group()
def cli():
    """命令行工具入口，包含多个子命令"""
    pass

@click.command()
@click.option('--start_id', default=-1, help='Start ID for processing')
@click.option('--limit', default=50, help='Limit the number of records to process')
def preprocess(start_id, limit):
    """执行 KG 特征预处理"""
    kg_llm_service.get_all_path_to_vdb_new(in_start_id=start_id, limit=limit)
    print(f"KG 特征预处理已启动，start_id={start_id}，limit={limit}")


# 将子命令添加到主命令组中
cli.add_command(preprocess)

if __name__ == "__main__":
    cli()