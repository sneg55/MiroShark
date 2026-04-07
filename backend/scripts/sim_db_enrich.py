"""
Database context enrichment helpers.

These functions read related rows from the SQLite simulation DB to add
human-readable context (post content, author names, etc.) to action records.
"""

from typing import Dict, Any, Optional


def _get_post_info(
    cursor,
    post_id: int,
    agent_names: Dict[int, str],
) -> Optional[Dict[str, str]]:
    """Return {'content': str, 'author_name': str} for a post, or None."""
    try:
        cursor.execute(
            """
            SELECT p.content, p.user_id, u.agent_id
            FROM post p
            LEFT JOIN user u ON p.user_id = u.user_id
            WHERE p.post_id = ?
            """,
            (post_id,),
        )
        row = cursor.fetchone()
        if row:
            content = row[0] or ''
            user_id = row[1]
            agent_id = row[2]

            author_name = ''
            if agent_id is not None and agent_id in agent_names:
                author_name = agent_names[agent_id]
            elif user_id:
                cursor.execute(
                    "SELECT name, user_name FROM user WHERE user_id = ?", (user_id,)
                )
                user_row = cursor.fetchone()
                if user_row:
                    author_name = user_row[0] or user_row[1] or ''

            return {'content': content, 'author_name': author_name}
    except Exception:
        pass
    return None


def _get_user_name(
    cursor,
    user_id: int,
    agent_names: Dict[int, str],
) -> Optional[str]:
    """Return the display name for a user_id, or None."""
    try:
        cursor.execute(
            "SELECT agent_id, name, user_name FROM user WHERE user_id = ?", (user_id,)
        )
        row = cursor.fetchone()
        if row:
            agent_id, name, user_name = row
            if agent_id is not None and agent_id in agent_names:
                return agent_names[agent_id]
            return name or user_name or ''
    except Exception:
        pass
    return None


def _get_comment_info(
    cursor,
    comment_id: int,
    agent_names: Dict[int, str],
) -> Optional[Dict[str, str]]:
    """Return {'content': str, 'author_name': str} for a comment, or None."""
    try:
        cursor.execute(
            """
            SELECT c.content, c.user_id, u.agent_id
            FROM comment c
            LEFT JOIN user u ON c.user_id = u.user_id
            WHERE c.comment_id = ?
            """,
            (comment_id,),
        )
        row = cursor.fetchone()
        if row:
            content = row[0] or ''
            user_id = row[1]
            agent_id = row[2]

            author_name = ''
            if agent_id is not None and agent_id in agent_names:
                author_name = agent_names[agent_id]
            elif user_id:
                cursor.execute(
                    "SELECT name, user_name FROM user WHERE user_id = ?", (user_id,)
                )
                user_row = cursor.fetchone()
                if user_row:
                    author_name = user_row[0] or user_row[1] or ''

            return {'content': content, 'author_name': author_name}
    except Exception:
        pass
    return None


def enrich_action_context(
    cursor,
    action_type: str,
    action_args: Dict[str, Any],
    agent_names: Dict[int, str],
) -> None:
    """
    Mutate action_args in-place to add human-readable context.

    Adds post/comment content, author names, and target user names where
    relevant. Errors are swallowed so enrichment never breaks the main flow.
    """
    try:
        if action_type in ('LIKE_POST', 'DISLIKE_POST'):
            post_id = action_args.get('post_id')
            if post_id:
                info = _get_post_info(cursor, post_id, agent_names)
                if info:
                    action_args['post_content'] = info['content']
                    action_args['post_author_name'] = info['author_name']

        elif action_type == 'REPOST':
            new_post_id = action_args.get('new_post_id')
            if new_post_id:
                cursor.execute(
                    "SELECT original_post_id FROM post WHERE post_id = ?", (new_post_id,)
                )
                row = cursor.fetchone()
                if row and row[0]:
                    info = _get_post_info(cursor, row[0], agent_names)
                    if info:
                        action_args['original_content'] = info['content']
                        action_args['original_author_name'] = info['author_name']

        elif action_type == 'QUOTE_POST':
            quoted_id = action_args.get('quoted_id')
            new_post_id = action_args.get('new_post_id')
            if quoted_id:
                info = _get_post_info(cursor, quoted_id, agent_names)
                if info:
                    action_args['original_content'] = info['content']
                    action_args['original_author_name'] = info['author_name']
            if new_post_id:
                cursor.execute(
                    "SELECT quote_content FROM post WHERE post_id = ?", (new_post_id,)
                )
                row = cursor.fetchone()
                if row and row[0]:
                    action_args['quote_content'] = row[0]

        elif action_type == 'FOLLOW':
            follow_id = action_args.get('follow_id')
            if follow_id:
                cursor.execute(
                    "SELECT followee_id FROM follow WHERE follow_id = ?", (follow_id,)
                )
                row = cursor.fetchone()
                if row:
                    name = _get_user_name(cursor, row[0], agent_names)
                    if name:
                        action_args['target_user_name'] = name

        elif action_type == 'MUTE':
            target_id = action_args.get('user_id') or action_args.get('target_id')
            if target_id:
                name = _get_user_name(cursor, target_id, agent_names)
                if name:
                    action_args['target_user_name'] = name

        elif action_type in ('LIKE_COMMENT', 'DISLIKE_COMMENT'):
            comment_id = action_args.get('comment_id')
            if comment_id:
                info = _get_comment_info(cursor, comment_id, agent_names)
                if info:
                    action_args['comment_content'] = info['content']
                    action_args['comment_author_name'] = info['author_name']

        elif action_type == 'CREATE_COMMENT':
            post_id = action_args.get('post_id')
            if post_id:
                info = _get_post_info(cursor, post_id, agent_names)
                if info:
                    action_args['post_content'] = info['content']
                    action_args['post_author_name'] = info['author_name']

    except Exception as e:
        print(f"Failed to enrich action context: {e}")
