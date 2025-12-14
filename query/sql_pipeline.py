# sql_pipeline.py

import os
import json
import re
import httpx
import sqlite3
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# -------------------------------
# Load environment
# -------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY environment variable not set.")
# -------------------------------
# Initialize OpenAI Client
# -------------------------------
verify_path = os.environ.get("REQUESTS_CA_BUNDLE", True)

client = OpenAI(
    api_key=OPENAI_API_KEY,
    http_client=httpx.Client(verify=verify_path)
)


# -------------------------------
# NFL Teams
# -------------------------------
NFL_TEAMS = [
    "ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN",
    "DET","GB","HOU","IND","JAX","KC","LV","LAC","LAR","MIA",
    "MIN","NE","NO","NYG","NYJ","PHI","PIT","SEA","SF","TB",
    "TEN","WAS"
]

team_list = ", ".join(NFL_TEAMS)

# -------------------------------
# Helper: Normalize CJ → C.J.
# -------------------------------
def normalize_player_name(name: str) -> str:
    if name is None:
        return None

    parts = name.split()
    if len(parts) > 1 and len(parts[0]) in {2, 3} and parts[0].isupper():
        initials = ".".join(parts[0]) + "."
        return f"{initials} {' '.join(parts[1:])}"

    return name


# =====================================================
# STEP 1: Intent + Entity Classification
# =====================================================
def classify_intent_and_extract_entities(question: str) -> dict:
    prompt = f"""
You are a classifier and entity extractor for a sports assistant focused on the Philadelphia Eagles.

You MUST return JSON in exactly the following structure:
{{
  "intent": "player_season | player_week | team_game | team_season",
  "entities": {{
    "season": integer or null,
    "week": integer or null,
    "player": string or null,
    "opponent": string or null
  }}
}}

Do NOT return JSON that is missing the "intent" or "entities" fields.

---

Your job is to:
1. Classify the question into one of these intents:
   - "player_season": full-season stats for a specific player (e.g. total TDs in 2023)
   - "player_week": weekly or opponent-specific stats for a player (e.g. vs Dallas or Week 9)
   - "team_game": team performance in a specific game or week
   - "team_season": team-wide totals for an entire season

2. Extract the following fields inside the "entities" object:
   - "season": integer (e.g. 2023) or null
   - "week": integer (e.g. 9) or null
   - "player": string (e.g. "Jalen Hurts" or "A.J. Brown") or null
   - "opponent": NFL team code (e.g. "DAL") or null

Guidelines:
- PHI should never be extracted as the opponent — the team is always the Eagles.
- Only extract opponents that are in this list: {team_list}
- Preserve initials in names like "A.J. Brown", "C.J. Gardner-Johnson"
- Return valid JSON only — no explanations, no formatting, no markdown fences.

User question:
{question}
"""


    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200
        )

        raw = response.choices[0].message.content
        #print("RAW MODEL OUTPUT:", repr(raw))

        # Strip code fences if present
        if raw.startswith("```"):
            raw = re.sub(r"```(json)?", "", raw).strip("` \n")

        # Parse JSON
        result = json.loads(raw)

        # ----------------------------------------------------
        # 🔥 FIX: Convert flat structure → nested structure
        # ----------------------------------------------------
        if "entities" not in result:
            result = {
                "intent": result.get("intent", "team_season"),
                "entities": {
                    "season": result.get("season"),
                    "week": result.get("week"),
                    "player": result.get("player"),
                    "opponent": result.get("opponent")
                }
            }

        entities = result["entities"]

        # ----------------------------------------------------
        # Normalize opponent field
        # ----------------------------------------------------
        opp = entities.get("opponent")
        if opp:
            opp = opp.upper()
            entities["opponent"] = opp if opp in NFL_TEAMS else None
        else:
            entities["opponent"] = None

        # ----------------------------------------------------
        # Normalize player name (AJ → A.J.)
        # ----------------------------------------------------
        entities["player"] = normalize_player_name(entities.get("player"))

        # ----------------------------------------------------
        # Validate intent
        # ----------------------------------------------------
        valid_intents = {"player_week", "player_season", "team_game", "team_season"}
        if result.get("intent") not in valid_intents:
            result["intent"] = "team_season"

        # Return final normalized result
        return {
            "intent": result["intent"],
            "entities": entities
        }

    except Exception as e:
        print("JSON PARSE ERROR:", e)
        return {
            "intent": "team_season",
            "entities": {
                "season": None,
                "week": None,
                "player": None,
                "opponent": None
            }
        }



# =====================================================
# STEP 2: SQL Builder
# =====================================================

def sql_rules(team_list: str) -> str:    
    return (
            "- Use ONLY the extracted entities above when building the WHERE clause.\n"
            "- NEVER pull team names or player names from the natural language question.\n"
            "- Use player_display_name for player filters.\n"
            "- Do NOT filter by team since all teams are PHI\n" 
            "- If the user asks for total 'yards' for a player, include both `passing_yards` and `rushing_yards` if available.\n"
            "- If the user asks about a change over be sure to include relevant time fields like year or week\n"
            "- If including filters for position, only use standard abbreviations (e.g., 'QB', 'WR', 'RB').\n"
            "- Do NOT include any field in the WHERE clause that is not listed in the table schema.\n"
            f"- Only extract opponents that are in this list: {team_list}"
            "- If per game stats are requested for a season, remember there are per game stats available in the games and seasons tables. They will have per_game in the field name.\n"
            "- Only the games table has an opponent field. If opponent level aggregation is needed it must be done in the games table.\n"
            "- In the games table, the win and loss fields are boolean TRUE/FALSE values.\n"
                        "- If stats are requested for a player in a specific game, use the player_week table.\n"
            "- If stats are requested for a player in a specific season, use the player_season table.\n"
            "- When querying the players table, season_type field should be 'REG+POST' unless regular-season(REG) or post-season(POST) stats are specifically requested.\n"
            "- If a question regarding when an event happened (e.g., highest yards in a game), be sure to include as year and week fields to identify the specific game.\n"
            )

            

sql_rules_text = sql_rules(team_list)


def select_table(intent: str) -> str:
    return {
        "player_week": "player_week",
        "player_season": "players",
        "team_game": "games",
        "team_season": "seasons"
    }.get(intent, "seasons")


def get_table_schema(db_path: str, table_name: str) -> str:
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        schema_lines = [f"{col[1]} ({col[2]})" for col in columns]
        return "\n".join(schema_lines)


def build_sql_prompt(question: str, schema: str, table: str, intent: str, entities: dict) -> str:
    season = entities.get("season")
    week = entities.get("week")
    player = entities.get("player")
    opponent = entities.get("opponent")

    # Build structured entity hint for LLM
    entity_block = f"""
EXTRACTED ENTITIES (use ONLY these values):
- season: {season}
- week: {week}
- player: {player}
- opponent: {opponent}
"""

    needs_aggregation = False
    if intent in ("player_week", "player_season"):
        if opponent and not week:
            needs_aggregation = True
        if week is None:
            needs_aggregation = True

    aggregation_rule = ""
    if needs_aggregation and table == "player_week":
        aggregation_rule = (
            "- If multiple rows match (e.g., multiple weeks vs same opponent), "
            "use SUM() for numeric fields.\n"
        )
    
    prompt = f"""
You are a data assistant that writes valid SQLite SQL queries.

TABLE NAME: {table}
SCHEMA:
{schema}

{entity_block}

Rules:
{sql_rules_text}

{aggregation_rule}- Return ONLY the SQL query with no commentary.

User question:
{question}
"""

    #print("🟩 SQL Prompt:\n", prompt)
    return prompt



def generate_sql_query(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=200
    )

    sql = response.choices[0].message.content

    # Clean up markdown-style code block if present
    if sql.startswith("```"):
        sql = re.sub(r"```(sql)?", "", sql).strip("` \n")

    #print("🟦 FINAL SQL QUERY:\n", sql)
    return sql


# =====================================================
# STEP 3: SQL Execution
# =====================================================
def run_sql_query(db_path: str, query: str):
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn)


# =====================================================
# STEP 4: Reflection
# =====================================================
def reflect_query_results(user_question: str, sql: str, query_results) -> str:
    """
    Reflects on the SQL and results, and determines whether the output answers the user’s question.
    """

    if hasattr(query_results, "to_string"):
        results_str = query_results.to_string(index=False)
    else:
        results_str = str(query_results)

    reflect_prompt = f"""You are a data assistant that evaluates whether a SQL query correctly answered a user's question.

Instructions:
- Look at the original user question.
- Examine the SQL query that was used to answer it.
- Review the returned results.
- Then determine:
  1. Did the SQL query match the user’s intent?
  2. Were the correct filters, fields, and aggregations used?
  3. Were all given rules followed?
  4. Do the results look reasonable?
- If correct, write a brief natural language answer.
- If incorrect, explain what went wrong (e.g. wrong column, filter, aggregation, etc.)

Rules:
    -filtering for team='PHI' is not necessary since all data is for the Eagles.

USER QUESTION:
{user_question}

GENERATED SQL:
{sql}

Rules:
{sql_rules_text}

QUERY RESULTS:
{results_str}
"""

    #print("🧠 Reflection Prompt:\n", reflect_prompt)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": reflect_prompt}],
        temperature=0,
        max_tokens=300
    )

    summary = response.choices[0].message.content.strip()
    return summary


# =====================================================
# STEP 5: Regeneration
# =====================================================
def build_regenerated_sql(user_question: str, reflection: str, previous_sql: str,
                          schema: str, table: str, intent: str, entities: dict) -> str:
    regen_prompt = f"""
You are a SQL assistant. Your job is to revise a previous SQL query that failed to correctly answer a user's question.

🟨 Original User Question:
{user_question}

🟥 Identified Problem with Previous SQL:
{reflection}

🧾 Previous SQL:
{previous_sql}

📘 Table Information:
- Table name: {table}
- Intent: {intent}
- Extracted entities: {json.dumps(entities, indent=2)}
- Table schema:
{schema}

Rules:
{sql_rules_text}
🎯 Goal:
Fix the SQL so it accurately answers the user’s question and follows all rules. Only return valid SQLite SQL — no commentary or markdown formatting.
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": regen_prompt}],
        temperature=0,
        max_tokens=200
    )

    new_sql = response.choices[0].message.content

    # Clean up markdown-style code block if present
    if new_sql.startswith("```"):
      new_sql = re.sub(r"```(sql)?", "", new_sql).strip("` \n")

    #print("🟪 REVISED SQL QUERY:\n", new_sql)
    return new_sql

def should_regenerate(reflection_summary: str, df_result: pd.DataFrame) -> bool:
    # 1. If empty result, almost always regenerate
    if df_result.empty:
        return True

    # 2. If reflection says it failed
    failure_keywords = [
        "did not successfully answer",
        "did not address",
        "does not address",
        "did not correctly",
        "does not correctly",
        "incorrectly",
        "failed to",
        "no data",
        "returned None",
        "no records found",
        "issue with the query",
        "mismatch",
        "missing"
    ]
    if any(kw in reflection_summary.lower() for kw in failure_keywords):
        return True

    return False
# =====================================================
# STEP 6: WRAPPED PIPELINE
# =====================================================
def run_query_pipeline(question: str):
    from paths import SQLITE_DB_PATH

    parsed = classify_intent_and_extract_entities(question)
    intent = parsed["intent"]
    entities = parsed["entities"]

    table = select_table(intent)
    schema = get_table_schema(SQLITE_DB_PATH, table)

    prompt = build_sql_prompt(question, schema, table, intent, entities)
    sql = generate_sql_query(prompt)

    df_result = run_sql_query(SQLITE_DB_PATH, sql)
    reflection_summary = reflect_query_results(question, sql, df_result)

    regenerated_sql = None
    if should_regenerate(reflection_summary, df_result):
        regenerated_sql = build_regenerated_sql(
            question, reflection_summary, sql, schema, table, intent, entities
        )
        df_result = run_sql_query(SQLITE_DB_PATH, regenerated_sql)

    return {
        "question": question,
        "intent": intent,
        "entities": entities,
        "table": table,
        "original_sql": sql,
        "regenerated_sql": regenerated_sql,
        "result": df_result,
        "reflection": reflection_summary
    }


# =====================================================
# OPTIONAL: TEST RUNS
# =====================================================
if __name__ == "__main__":

    question = "How many rushing yards per game did the eagles have in 2024?"

    output = run_query_pipeline(question)

    print("🟨 USER QUESTION:", output["question"])
    print("🧠 Intent:", output["intent"])
    print("🧩 Entities:", output["entities"])
    print("📄 Table:", output["table"])
    print("🟦 Original SQL:\n", output["original_sql"])
    print("🧠 Reflection:\n", output["reflection"])
    print("🔁 Regenerated SQL:\n", output["regenerated_sql"])
    print("📊 Result:\n", output["result"])