import argparse
import asyncio
from dataclasses import asdict
import json
import logging

from app.config import get_settings
from app.db.pool import create_database_pool, close_database_pool
from app.publication.auto_service import run_auto_publication


async def main(*, dry_run: bool) -> int:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    pool = await create_database_pool(settings)
    try:
        result = await run_auto_publication(pool, settings=settings, dry_run=dry_run)
        print(json.dumps(asdict(result), ensure_ascii=False), flush=True)
        return 0 if result.published or result.skipped_reason else 1
    finally:
        await close_database_pool(pool)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Публикация сегодняшнего поста без реакции к 12:00 UTC')
    parser.add_argument('--dry-run', action='store_true', help='Только проверить выбор и файлы, без записей и Telegram')
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(dry_run=args.dry_run)))
