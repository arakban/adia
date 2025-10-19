# Task 4: Text File Ingestion

## Problem Statement

Your research group is looking at the affects of covid on government spending. You've found a kaggle dataset that has collected Covid-19 research papers, and the group would like to use the number of papers published by country as an indicator of the level of funded research. 

The group has decided to look for country name in the title and to map author emails to country (via a separate process). So you will need to store this metadata in a database structure.

## Task Requirements

You are provided with a small subset of the CORD19 dataset in a zipfile. Design a data schema to hold the necessary data, and write a python function that will upload them into your table structure. 

**Use the sqlite in memory database (syntax provided in the code blocks).**

---

## Key Requirements

1. **Data Extraction**:
   - Process Covid-19 research papers from CORD19 dataset
   - Extract country names from paper titles
   - Extract author email addresses
   - Store metadata for analysis

2. **Database Design**:
   - Design schema for research paper metadata
   - Use SQLite in-memory database
   - Follow provided syntax in code blocks

3. **Python Function**:
   - Read from provided zipfile
   - Parse paper metadata
   - Insert into database tables
   - Handle text processing for country extraction

---

## Domain Context

**CORD19 Dataset**: 
- Large collection of Covid-19 research papers
- Metadata includes: title, authors, emails, publication info
- Used for research funding analysis by country
- Requires text processing for country name extraction

## Expected Data Elements

- Paper ID/identifier
- Paper title
- Author information
- Author emails
- Country (extracted from title or email)
- Publication date/metadata

## Implementation Approach

### Step 1: SQLite Setup
```python
import sqlite3
import pandas as pd
import zipfile

# Create in-memory database
conn = sqlite3.connect(':memory:')
cursor = conn.cursor()
```

### Step 2: Schema Design
Consider tables for:
- Papers (paper_id, title, publication_date)
- Authors (author_id, name, email, country)
- Paper_Authors (linking table)
- Countries (country metadata)

### Step 3: Text Processing
- Extract country names from titles (regex/NLP)
- Parse email domains for country mapping
- Handle multiple authors per paper
- Clean and normalize country names

### Step 4: Data Loading
- Extract files from zipfile
- Parse paper metadata (likely JSON or CSV)
- Insert into database tables
- Handle duplicates and data quality issues