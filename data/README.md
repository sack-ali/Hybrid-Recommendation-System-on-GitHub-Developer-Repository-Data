# Data

This folder expects the three CSV files from the GitHub Bipartite Graph Dataset.

## Expected files

```
data/
├── developers.csv
├── repositories.csv
└── edgelist.csv
```

## Schema

### `developers.csv`

| Column | Type | Description |
|--------|------|-------------|
| `dev_id` | str / int | Unique developer identifier |
| `Followers` | int | Number of followers |
| `Following` | int | Number of accounts followed |
| `Public Repositories` | int | Count of public repos |
| `starredRepoCount` | int | Number of starred repositories |
| `yearly_contributions` | int | Total contributions in the past year |
| `Bio` | str | Free-text bio / self description |

### `repositories.csv`

| Column | Type | Description |
|--------|------|-------------|
| `repo_id` | str / int | Unique repository identifier |
| `repo_name` | str | Repository name |
| `owner_username` | str | Owner GitHub username |
| `description` | str | Short repo description |
| `readme` | str | Extracted README text |
| `topics` | list / str | Repository topics |
| `languages` | list / str | Programming languages used |
| `stargazers_count` | int | Stars |
| `forks_count` | int | Forks |
| `watching` | int | Watchers |
| `contributors_count` | int | Number of contributors |
| `commits_count` | int | Total commits |
| `open_issues_count` | int | Open issues |
| `size` | int | Repo size |

### `edgelist.csv`

| Column | Type | Description |
|--------|------|-------------|
| `dev_id` | str / int | Developer id |
| `repo_id` | str / int | Repository id |
| `isForked` | 0/1 | Whether the developer forked the repo |
| `isTopContributor` | 0/1 | Whether the developer is a top contributor |

## Notes

- The CSVs are **not committed** to the repository — see `.gitignore`.
- The pipeline is tolerant to missing values: numeric columns are coerced and filled with `0`, and text columns are filled with empty strings.
