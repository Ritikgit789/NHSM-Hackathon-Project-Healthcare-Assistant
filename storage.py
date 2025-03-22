from agno.storage.agent.sqlite import SqliteAgentStorage

def load_storage() -> SqliteAgentStorage:
    storage= SqliteAgentStorage(
        table_name="agent_sessions", 
        db_file="tmp/agent_storage.db"
    )
    return storage