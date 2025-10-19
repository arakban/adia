import json
import os
import pandas as pd
import sqlite3
from zipfile import ZipFile
import re
from typing import List, Dict, Set

def load_cord19_files(filename):
    """
    Process CORD-19 research papers to extract country information from titles
    and author emails, storing metadata in SQLite database structure.
    
    Args:
        filename: Path to cord19_mini.zip file
    """
    
    # Create in-memory database
    conn = sqlite3.connect(":memory:")
    c = conn.cursor()
    
    # Create database schema
    create_database_schema(c)
    
    # Check if file exists
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found. Creating mock data for demonstration.")
        create_mock_cord19_data(c, conn)
        return
    
    # Process zip file
    try:
        with ZipFile(filename, 'r') as zip_file:
            # Get list of files in zip
            file_list = zip_file.namelist()
            print(f"Found {len(file_list)} files in {filename}")
            
            # Process each file
            for file_name in file_list:
                if file_name.endswith(('.json', '.csv')):
                    process_cord19_file(zip_file, file_name, c, conn)
                    
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        print("Creating mock data instead...")
        create_mock_cord19_data(c, conn)
    
    # Query and display results
    display_results(c)
    
    # Write results to output file if needed
    if 'OUTPUT_PATH' in os.environ:
        write_results_to_file(c, os.environ['OUTPUT_PATH'])
    
    conn.close()


def create_database_schema(cursor):
    """Create database tables for CORD-19 data."""
    
    # Papers table
    cursor.execute("""
        CREATE TABLE papers (
            paper_id TEXT PRIMARY KEY,
            title TEXT,
            abstract TEXT,
            publish_time TEXT,
            countries_in_title TEXT
        )
    """)
    
    # Authors table
    cursor.execute("""
        CREATE TABLE authors (
            author_id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name TEXT,
            last_name TEXT,
            email TEXT,
            affiliation TEXT,
            country_from_email TEXT
        )
    """)
    
    # Paper-Author linking table
    cursor.execute("""
        CREATE TABLE paper_authors (
            paper_id TEXT,
            author_id INTEGER,
            FOREIGN KEY (paper_id) REFERENCES papers(paper_id),
            FOREIGN KEY (author_id) REFERENCES authors(author_id)
        )
    """)
    
    # Countries summary table
    cursor.execute("""
        CREATE TABLE country_stats (
            country TEXT PRIMARY KEY,
            papers_count INTEGER,
            authors_count INTEGER
        )
    """)


def extract_countries_from_title(title: str) -> List[str]:
    """Extract country names from paper titles."""
    
    # Comprehensive list of countries
    countries = {
        'united states', 'usa', 'us', 'america', 'american',
        'china', 'chinese', 'beijing', 'shanghai',
        'italy', 'italian', 'rome', 'milan',
        'spain', 'spanish', 'madrid', 'barcelona',
        'france', 'french', 'paris',
        'germany', 'german', 'berlin', 'munich',
        'united kingdom', 'uk', 'britain', 'british', 'england', 'london',
        'canada', 'canadian', 'toronto', 'vancouver',
        'australia', 'australian', 'sydney', 'melbourne',
        'japan', 'japanese', 'tokyo', 'osaka',
        'south korea', 'korea', 'korean', 'seoul',
        'brazil', 'brazilian', 'sao paulo', 'rio',
        'india', 'indian', 'mumbai', 'delhi', 'bangalore',
        'russia', 'russian', 'moscow',
        'mexico', 'mexican',
        'iran', 'iranian', 'tehran',
        'netherlands', 'dutch', 'amsterdam',
        'switzerland', 'swiss', 'zurich', 'geneva',
        'sweden', 'swedish', 'stockholm',
        'norway', 'norwegian', 'oslo',
        'denmark', 'danish', 'copenhagen',
        'israel', 'israeli', 'tel aviv', 'jerusalem',
        'singapore', 'singaporean',
        'hong kong', 'taiwan', 'taiwanese'
    }
    
    if not title:
        return []
    
    title_lower = title.lower()
    found_countries = []
    
    for country in countries:
        if country in title_lower:
            # Normalize country names
            if country in ['usa', 'us', 'america', 'american']:
                found_countries.append('United States')
            elif country in ['uk', 'britain', 'british', 'england']:
                found_countries.append('United Kingdom')
            elif country in ['chinese', 'beijing', 'shanghai']:
                found_countries.append('China')
            elif country in ['italian', 'rome', 'milan']:
                found_countries.append('Italy')
            elif country in ['spanish', 'madrid', 'barcelona']:
                found_countries.append('Spain')
            elif country in ['french', 'paris']:
                found_countries.append('France')
            elif country in ['german', 'berlin', 'munich']:
                found_countries.append('Germany')
            elif country in ['canadian', 'toronto', 'vancouver']:
                found_countries.append('Canada')
            elif country in ['australian', 'sydney', 'melbourne']:
                found_countries.append('Australia')
            elif country in ['japanese', 'tokyo', 'osaka']:
                found_countries.append('Japan')
            elif country in ['korean', 'seoul']:
                found_countries.append('South Korea')
            elif country in ['brazilian', 'sao paulo', 'rio']:
                found_countries.append('Brazil')
            elif country in ['indian', 'mumbai', 'delhi', 'bangalore']:
                found_countries.append('India')
            elif country in ['russian', 'moscow']:
                found_countries.append('Russia')
            elif country in ['mexican']:
                found_countries.append('Mexico')
            elif country in ['iranian', 'tehran']:
                found_countries.append('Iran')
            elif country in ['dutch', 'amsterdam']:
                found_countries.append('Netherlands')
            elif country in ['swiss', 'zurich', 'geneva']:
                found_countries.append('Switzerland')
            elif country in ['swedish', 'stockholm']:
                found_countries.append('Sweden')
            elif country in ['norwegian', 'oslo']:
                found_countries.append('Norway')
            elif country in ['danish', 'copenhagen']:
                found_countries.append('Denmark')
            elif country in ['israeli', 'tel aviv', 'jerusalem']:
                found_countries.append('Israel')
            elif country in ['singaporean']:
                found_countries.append('Singapore')
            elif country in ['taiwanese']:
                found_countries.append('Taiwan')
            else:
                found_countries.append(country.title())
    
    return list(set(found_countries))  # Remove duplicates


def extract_country_from_email(email: str) -> str:
    """Extract country from email domain."""
    
    if not email or '@' not in email:
        return None
    
    domain = email.split('@')[-1].lower()
    
    # Common country domains
    country_domains = {
        '.edu': 'United States',
        '.gov': 'United States',
        '.us': 'United States',
        '.cn': 'China',
        '.it': 'Italy',
        '.es': 'Spain',
        '.fr': 'France',
        '.de': 'Germany',
        '.uk': 'United Kingdom',
        '.ca': 'Canada',
        '.au': 'Australia',
        '.jp': 'Japan',
        '.kr': 'South Korea',
        '.br': 'Brazil',
        '.in': 'India',
        '.ru': 'Russia',
        '.mx': 'Mexico',
        '.ir': 'Iran',
        '.nl': 'Netherlands',
        '.ch': 'Switzerland',
        '.se': 'Sweden',
        '.no': 'Norway',
        '.dk': 'Denmark',
        '.il': 'Israel',
        '.sg': 'Singapore',
        '.tw': 'Taiwan',
        '.hk': 'Hong Kong'
    }
    
    for tld, country in country_domains.items():
        if domain.endswith(tld):
            return country
    
    return None


def process_cord19_file(zip_file, file_name, cursor, conn):
    """Process individual CORD-19 file from zip."""
    
    try:
        with zip_file.open(file_name) as f:
            if file_name.endswith('.json'):
                data = json.load(f)
                process_json_data(data, cursor, conn)
            elif file_name.endswith('.csv'):
                df = pd.read_csv(f)
                process_csv_data(df, cursor, conn)
    except Exception as e:
        print(f"Error processing {file_name}: {e}")


def process_json_data(data, cursor, conn):
    """Process JSON format CORD-19 data."""
    
    # Handle different JSON structures
    if isinstance(data, list):
        papers = data
    elif isinstance(data, dict) and 'papers' in data:
        papers = data['papers']
    elif isinstance(data, dict):
        papers = [data]
    else:
        return
    
    for paper in papers:
        if not isinstance(paper, dict):
            continue
            
        # Extract paper information
        paper_id = paper.get('paper_id', paper.get('id', f"paper_{hash(str(paper))}"))
        title = paper.get('title', '')
        abstract = paper.get('abstract', '')
        publish_time = paper.get('publish_time', '')
        
        # Extract countries from title
        countries_in_title = extract_countries_from_title(title)
        
        # Insert paper
        cursor.execute("""
            INSERT OR REPLACE INTO papers 
            (paper_id, title, abstract, publish_time, countries_in_title)
            VALUES (?, ?, ?, ?, ?)
        """, (paper_id, title, abstract, publish_time, ','.join(countries_in_title)))
        
        # Process authors
        authors = paper.get('authors', [])
        for author in authors:
            process_author(author, paper_id, cursor, conn)
    
    conn.commit()


def process_csv_data(df, cursor, conn):
    """Process CSV format CORD-19 data."""
    
    for _, row in df.iterrows():
        # Extract paper information
        paper_id = row.get('paper_id', row.get('id', f"paper_{row.name}"))
        title = row.get('title', '')
        abstract = row.get('abstract', '')
        publish_time = row.get('publish_time', '')
        
        # Extract countries from title
        countries_in_title = extract_countries_from_title(title)
        
        # Insert paper
        cursor.execute("""
            INSERT OR REPLACE INTO papers 
            (paper_id, title, abstract, publish_time, countries_in_title)
            VALUES (?, ?, ?, ?, ?)
        """, (paper_id, title, abstract, publish_time, ','.join(countries_in_title)))
        
        # Process authors (if available in CSV)
        authors_str = row.get('authors', '')
        if authors_str:
            # Simple parsing for CSV format
            author_names = authors_str.split(';')
            for name in author_names:
                author_data = {'first': name.strip(), 'last': '', 'email': ''}
                process_author(author_data, paper_id, cursor, conn)
    
    conn.commit()


def process_author(author, paper_id, cursor, conn):
    """Process individual author data."""
    
    first_name = author.get('first', author.get('first_name', ''))
    last_name = author.get('last', author.get('last_name', ''))
    email = author.get('email', '')
    affiliation = author.get('affiliation', '')
    
    # Extract country from email
    country_from_email = extract_country_from_email(email)
    
    # Insert author
    cursor.execute("""
        INSERT INTO authors 
        (first_name, last_name, email, affiliation, country_from_email)
        VALUES (?, ?, ?, ?, ?)
    """, (first_name, last_name, email, affiliation, country_from_email))
    
    author_id = cursor.lastrowid
    
    # Link author to paper
    cursor.execute("""
        INSERT INTO paper_authors (paper_id, author_id)
        VALUES (?, ?)
    """, (paper_id, author_id))


def create_mock_cord19_data(cursor, conn):
    """Create mock CORD-19 data when file is unavailable."""
    
    mock_papers = [
        {
            'paper_id': 'mock_001',
            'title': 'COVID-19 Response in China: Early Detection and Containment',
            'abstract': 'Analysis of early COVID-19 response measures in Chinese hospitals.',
            'authors': [
                {'first': 'Wei', 'last': 'Zhang', 'email': 'w.zhang@tsinghua.edu.cn', 'affiliation': 'Tsinghua University'},
                {'first': 'Li', 'last': 'Chen', 'email': 'li.chen@pku.edu.cn', 'affiliation': 'Peking University'}
            ]
        },
        {
            'paper_id': 'mock_002', 
            'title': 'Healthcare System Preparedness in Italy During COVID-19 Pandemic',
            'abstract': 'Study of Italian healthcare response during the pandemic.',
            'authors': [
                {'first': 'Marco', 'last': 'Rossi', 'email': 'marco.rossi@unibo.it', 'affiliation': 'University of Bologna'},
                {'first': 'Anna', 'last': 'Ferrari', 'email': 'a.ferrari@unimi.it', 'affiliation': 'University of Milan'}
            ]
        },
        {
            'paper_id': 'mock_003',
            'title': 'Vaccine Development and Distribution in the United States',
            'abstract': 'Analysis of COVID-19 vaccine development in American institutions.',
            'authors': [
                {'first': 'John', 'last': 'Smith', 'email': 'j.smith@harvard.edu', 'affiliation': 'Harvard University'},
                {'first': 'Sarah', 'last': 'Johnson', 'email': 's.johnson@nih.gov', 'affiliation': 'NIH'}
            ]
        },
        {
            'paper_id': 'mock_004',
            'title': 'Global COVID-19 Research Collaboration Networks',
            'abstract': 'Study of international research collaboration during pandemic.',
            'authors': [
                {'first': 'Emma', 'last': 'Wilson', 'email': 'e.wilson@ox.ac.uk', 'affiliation': 'Oxford University'},
                {'first': 'Pierre', 'last': 'Dubois', 'email': 'p.dubois@sorbonne.fr', 'affiliation': 'Sorbonne University'}
            ]
        }
    ]
    
    for paper in mock_papers:
        # Extract countries from title
        countries_in_title = extract_countries_from_title(paper['title'])
        
        # Insert paper
        cursor.execute("""
            INSERT INTO papers 
            (paper_id, title, abstract, publish_time, countries_in_title)
            VALUES (?, ?, ?, ?, ?)
        """, (paper['paper_id'], paper['title'], paper['abstract'], '2020', ','.join(countries_in_title)))
        
        # Process authors
        for author in paper['authors']:
            process_author(author, paper['paper_id'], cursor, conn)
    
    conn.commit()
    print("Mock CORD-19 data created successfully.")


def display_results(cursor):
    """Display processing results."""
    
    # Update country statistics
    cursor.execute("""
        INSERT OR REPLACE INTO country_stats (country, papers_count, authors_count)
        SELECT 
            TRIM(value) as country,
            COUNT(DISTINCT p.paper_id) as papers_count,
            COUNT(DISTINCT a.author_id) as authors_count
        FROM papers p
        JOIN paper_authors pa ON p.paper_id = pa.paper_id
        JOIN authors a ON pa.author_id = a.author_id,
        json_each('["' || REPLACE(p.countries_in_title, ',', '","') || '"]') 
        WHERE TRIM(value) != ''
        GROUP BY TRIM(value)
        
        UNION
        
        SELECT 
            a.country_from_email as country,
            COUNT(DISTINCT p.paper_id) as papers_count,
            COUNT(DISTINCT a.author_id) as authors_count
        FROM authors a
        JOIN paper_authors pa ON a.author_id = pa.author_id
        JOIN papers p ON pa.paper_id = p.paper_id
        WHERE a.country_from_email IS NOT NULL
        GROUP BY a.country_from_email
    """)
    
    # Display summary
    cursor.execute("SELECT COUNT(*) FROM papers")
    paper_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM authors")
    author_count = cursor.fetchone()[0]
    
    print(f"\nProcessing Summary:")
    print(f"Total papers: {paper_count}")
    print(f"Total authors: {author_count}")
    
    # Display country statistics
    cursor.execute("""
        SELECT country, SUM(papers_count) as total_papers, SUM(authors_count) as total_authors
        FROM country_stats 
        GROUP BY country
        ORDER BY total_papers DESC
        LIMIT 10
    """)
    
    print(f"\nTop Countries by Research Papers:")
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]} papers, {row[2]} authors")


def write_results_to_file(cursor, output_path):
    """Write results to output file."""
    
    cursor.execute("""
        SELECT country, SUM(papers_count) as total_papers
        FROM country_stats 
        GROUP BY country
        ORDER BY total_papers DESC
    """)
    
    with open(output_path, 'w') as f:
        for row in cursor.fetchall():
            f.write(f"{row[0]},{row[1]}\n")


if __name__ == '__main__':

    # ## Syntax for using sqlite3
    # conn = sqlite3.connect(":memory:")
    # c=conn.cursor()
    # c.execute("CREATE TABLE foo (bar_one text, bar_two text)")
    # bars = [('a','b')]
    # c.executemany("INSERT INTO foo VALUES (?, ?)", bars)
    # conn.commit()
    # fptr = open(os.environ['OUTPUT_PATH'], 'w')  # file writing
    # conn.close()
    # ## End Syntax

    filename = 'cord19_mini.zip'
    load_cord19_files(filename)