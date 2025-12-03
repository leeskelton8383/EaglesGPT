import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

NARRATIVE_DIR = os.path.join(DATA_DIR, "narratives")
CHUNKS_JSONL_PATH = os.path.join(DATA_DIR, "chunks.jsonl")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "narrative_index.faiss")
METADATA_JSON_PATH = os.path.join(DATA_DIR, "narrative_metadata.json")
SQLITE_DB_PATH = os.path.join(DATA_DIR, "stats.sqlite")

PLAYERS_CSV = os.path.join(DATA_DIR, "stats_player_regpost_2020_2025.csv")
GAMES_CSV = os.path.join(DATA_DIR, "eagles_game_summary_by_week_with_srs_cumleague_z_rank_2020_2025.csv")
SEASONS_CSV = os.path.join(DATA_DIR, "eagles_season_summary_with_srs_cumleague_z_rank_2020_2025.csv")
PLAYER_WEEKS_CSV = os.path.join(DATA_DIR, "stats_player_week_2020_2025.csv")