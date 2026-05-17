# FRAC_KG
FRAC knowledge graph Streamlit app

## Streamlit secrets

This app requires the following Streamlit secrets:

```toml
NEO4J_URI = "neo4j+s://<host>"
NEO4J_USER = "<username>"
NEO4J_PASSWORD = "<password>"
```

Configure them in Streamlit Cloud under app settings before deploying. For local
development, place the same keys in `.streamlit/secrets.toml`; do not commit that
file.
