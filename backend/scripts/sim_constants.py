"""
Simulation constants: action type lists, IPC directory names, and action type maps.
"""

try:
    from wonderwall import ActionType
except ImportError:
    ActionType = None  # Allow import without wonderwall for tests/tooling

# Available Twitter actions (excluding INTERVIEW, which can only be triggered manually)
TWITTER_ACTIONS = [
    ActionType.CREATE_POST,
    ActionType.LIKE_POST,
    ActionType.REPOST,
    ActionType.FOLLOW,
    ActionType.DO_NOTHING,
    ActionType.QUOTE_POST,
] if ActionType else []

# Available Reddit actions (excluding INTERVIEW)
REDDIT_ACTIONS = [
    ActionType.LIKE_POST,
    ActionType.DISLIKE_POST,
    ActionType.CREATE_POST,
    ActionType.CREATE_COMMENT,
    ActionType.LIKE_COMMENT,
    ActionType.DISLIKE_COMMENT,
    ActionType.SEARCH_POSTS,
    ActionType.SEARCH_USER,
    ActionType.TREND,
    ActionType.REFRESH,
    ActionType.DO_NOTHING,
    ActionType.FOLLOW,
    ActionType.MUTE,
] if ActionType else []

# IPC directory / file names
IPC_COMMANDS_DIR = "ipc_commands"
IPC_RESPONSES_DIR = "ipc_responses"
ENV_STATUS_FILE = "env_status.json"

# Non-core action types to filter out (low analytical value)
FILTERED_ACTIONS = {'refresh', 'sign_up'}

# Action type mapping (database name -> standard name)
ACTION_TYPE_MAP = {
    'create_post': 'CREATE_POST',
    'like_post': 'LIKE_POST',
    'dislike_post': 'DISLIKE_POST',
    'repost': 'REPOST',
    'quote_post': 'QUOTE_POST',
    'follow': 'FOLLOW',
    'mute': 'MUTE',
    'create_comment': 'CREATE_COMMENT',
    'like_comment': 'LIKE_COMMENT',
    'dislike_comment': 'DISLIKE_COMMENT',
    'search_posts': 'SEARCH_POSTS',
    'search_user': 'SEARCH_USER',
    'trend': 'TREND',
    'do_nothing': 'DO_NOTHING',
    'interview': 'INTERVIEW',
}

# Polymarket action type mapping
POLYMARKET_ACTION_TYPE_MAP = {
    'browse_markets': 'BROWSE_MARKETS',
    'buy_shares': 'BUY_SHARES',
    'sell_shares': 'SELL_SHARES',
    'view_portfolio': 'VIEW_PORTFOLIO',
    'create_market': 'CREATE_MARKET',
    'comment_on_market': 'COMMENT_ON_MARKET',
    'do_nothing': 'DO_NOTHING',
    'sign_up': 'SIGN_UP',
}
