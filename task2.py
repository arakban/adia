#!/bin/python3

#
# Complete the 'get_company_csv_list' function below.
#
# You will be able to view any debug output via print().
#

import os
from typing import List
import requests
import time
import xml.etree.ElementTree as ET
from datetime import datetime

TEMP_API_KEY = "Kf1fmqa3XaGGGsh6wMw5OPlYgsHA1FTz"

def get_company_csv_list(from_article: str) -> List[str]:
    """
    Extract company information from news articles using Thomson Reuters APIs.
    
    Args:
        from_article: XML string containing article title and body
        
    Returns:
        List of CSV-formatted strings with company information
    """
    
    def parse_article_xml(xml_content: str):
        """Parse XML article to extract title and body."""
        try:
            root = ET.fromstring(xml_content)
            title = root.find('Title')
            body = root.find('Body')
            
            return {
                'title': title.text if title is not None and title.text else '',
                'body': body.text if body is not None and body.text else ''
            }
        except ET.ParseError as e:
            print(f"Invalid XML format: {e}")
            return {'title': '', 'body': ''}
    
    def call_intelligent_tags_api(xml_content: str):
        """Call the Intelligent Tags API to extract company entities."""
        headers = {
            'X-AG-Access-Token': TEMP_API_KEY,
            'Content-Type': 'text/xml',
            'outputFormat': 'application/json',
            'x-calais-selectiveTags': 'company'
        }
        
        # Rate limiting - wait 1 second
        time.sleep(1)
        
        try:
            response = requests.post(
                "https://api-eit.refinitiv.com/permid/calais", 
                headers=headers, 
                data=xml_content
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API call failed: {e}")
            return {}
    
    def extract_companies_from_response(api_response):
        """Extract company information from Intelligent Tags API response."""
        companies = []
        
        for key, value in api_response.items():
            if isinstance(value, dict) and value.get('_type') == 'Company':
                resolutions = value.get('resolutions', [])
                
                for resolution in resolutions:
                    if 'permid' in resolution:
                        company_info = {
                            'permid': resolution['permid'],
                            'name': resolution.get('name', ''),
                            'ticker': resolution.get('ticker') if resolution.get('ispublic') == 'true' else None
                        }
                        companies.append(company_info)
        
        return companies
    
    def get_ipo_date(permid: str):
        """Get IPO date for a company using PermId Info API."""
        url = f"https://permid.org/1-{permid}"
        headers = {
            'X-AG-Access-Token': TEMP_API_KEY
        }
        params = {
            'format': 'json-ld'
        }
        
        # Rate limiting - wait 1 second
        time.sleep(1)
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            ipo_date_str = data.get('hasIPODate')
            if ipo_date_str:
                # Parse ISO 8601 timestamp and convert to YYYY-MM-DD
                ipo_datetime = datetime.fromisoformat(ipo_date_str.replace('Z', '+00:00'))
                return ipo_datetime.strftime('%Y-%m-%d')
            
            return None
        except (requests.exceptions.RequestException, ValueError, KeyError):
            return None
    
    def format_output(companies):
        """Format company data as CSV strings sorted by PermId."""
        # Sort by PermId (numeric sort)
        companies_sorted = sorted(companies, key=lambda x: int(x['permid']))
        
        csv_lines = []
        for company in companies_sorted:
            permid = company['permid']
            name = f"'{company['name']}'"
            ticker = company['ticker'] if company['ticker'] else 'NULL'
            ipo_date = company.get('ipo_date', 'NULL')
            
            csv_line = f"{permid},{name},{ticker},{ipo_date}"
            csv_lines.append(csv_line)
        
        return csv_lines
    
    # Main processing logic
    try:
        # Step 1: Parse XML article
        article_data = parse_article_xml(from_article)
        
        # Step 2: Call Intelligent Tags API
        api_response = call_intelligent_tags_api(from_article)
        
        # Step 3: Extract company entities
        companies = extract_companies_from_response(api_response)
        
        # Step 4: Get IPO dates for each company
        for company in companies:
            company['ipo_date'] = get_ipo_date(company['permid'])
        
        # Step 5: Format output
        return format_output(companies)
        
    except Exception as e:
        print(f"Error processing article: {e}")
        return []


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    from_article = input()

    result = get_company_csv_list(from_article)

    fptr.write('\n'.join(result))
    fptr.write('\n')

    fptr.close()