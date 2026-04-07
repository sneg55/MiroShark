"""
Database fetch helpers for reading simulation action records.

Reads the SQLite trace table produced by Wonderwall and normalises
rows into the action-dict format used throughout the pipeline.
"""

import json
import os
import sqlite3
from typing import Any, Dict, List, Tuple

from sim_constants import ACTION_TYPE_MAP, FILTERED_ACTIONS, POLYMARKET_ACTION_TYPE_MAP
from sim_db_enrich import enrich_action_context


def fetch_new_actions_from_db(
    db_path: str,
    last_rowid: int,
    agent_names: Dict[int, str],
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Fetch new action records from the Wonderwall trace table.

    Uses rowid (not created_at) to track progress because Twitter and Reddit
    format created_at differently.

    Returns:
        (actions_list, new_last_rowid)
    """
    actions: List[Dict[str, Any]] = []
    new_last_rowid = last_rowid

    if not os.path.exists(db_path):
        return actions, new_last_rowid

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT rowid, user_id, action, info
            FROM trace
            WHERE rowid > ?
            ORDER BY rowid ASC
            """,
            (last_rowid,),
        )

        for rowid, user_id, action, info_json in cursor.fetchall():
            new_last_rowid = rowid

            if action in FILTERED_ACTIONS:
                continue

            try:
                action_args = json.loads(info_json) if info_json else {}
            except json.JSONDecodeError:
                action_args = {}

            # Keep only key fields (full content, no truncation)
            simplified: Dict[str, Any] = {}
            for key in ('content', 'post_id', 'comment_id', 'quoted_id',
                        'new_post_id', 'follow_id', 'query', 'like_id', 'dislike_id'):
                if key in action_args:
                    simplified[key] = action_args[key]

            action_type = ACTION_TYPE_MAP.get(action, action.upper())
            enrich_action_context(cursor, action_type, simplified, agent_names)

            actions.append({
                'agent_id': user_id,
                'agent_name': agent_names.get(user_id, f'Agent_{user_id}'),
                'action_type': action_type,
                'action_args': simplified,
            })

        conn.close()
    except Exception as e:
        print(f"Failed to read actions from database: {e}")

    return actions, new_last_rowid


def fetch_polymarket_actions_from_db(
    db_path: str,
    last_rowid: int,
    agent_names: Dict[int, str],
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Fetch new Polymarket actions from the trace table.

    Same pattern as fetch_new_actions_from_db but adapted for
    Polymarket's action names and richer trade data.
    """
    actions: List[Dict[str, Any]] = []
    new_last_rowid = last_rowid

    if not os.path.exists(db_path):
        return actions, new_last_rowid

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT rowid, user_id, action, info
            FROM trace
            WHERE rowid > ?
            ORDER BY rowid ASC
            """,
            (last_rowid,),
        )

        for rowid, user_id, action, info_json in cursor.fetchall():
            new_last_rowid = rowid

            if action in ('sign_up',):
                continue

            try:
                action_args = json.loads(info_json) if info_json else {}
            except json.JSONDecodeError:
                action_args = {}

            action_type = POLYMARKET_ACTION_TYPE_MAP.get(action, action.upper())

            actions.append({
                'agent_id': user_id,
                'agent_name': agent_names.get(user_id, f'Agent_{user_id}'),
                'action_type': action_type,
                'action_args': action_args,
            })

        conn.close()
    except Exception as e:
        print(f"Failed to read Polymarket actions from database: {e}")

    return actions, new_last_rowid
