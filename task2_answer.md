# ADIA Task 2: Tickers from Articles - Solution

## Implementation Overview

Complete solution for extracting company information from news articles using Thomson Reuters APIs, implemented using the provided skeleton code structure.

## Core Functionality

**Main Function**: `get_company_csv_list(from_article: str) -> List[str]`

### Internal Helper Functions:

1. **`parse_article_xml(xml_content)`** - Extracts title and body from XML
2. **`call_intelligent_tags_api(xml_content)`** - Calls Calais API with rate limiting
3. **`extract_companies_from_response(api_response)`** - Parses company entities
4. **`get_ipo_date(permid)`** - Retrieves IPO dates via PermId API
5. **`format_output(companies)`** - Formats as CSV strings

## Technical Implementation

### API Configuration
- **Calais API**: `https://api-eit.refinitiv.com/permid/calais`
- **PermId API**: `https://permid.org/1-{PERM_ID}`
- **Token**: `Kf1fmqa3XaGGGsh6wMw5OPlYgsHA1FTz`
- **Rate Limiting**: 1 second wait between ALL API calls

### Data Processing Pipeline

1. **XML Parsing**: Extract title/body using ElementTree
2. **Entity Extraction**: POST to Calais API with `x-calais-selectiveTags: company`
3. **Company Filtering**: Only include entities with `resolutions` array
4. **Ticker Validation**: Only extract tickers where `ispublic == "true"`
5. **IPO Date Retrieval**: GET from PermId API for each company
6. **Date Formatting**: Convert ISO 8601 to YYYY-MM-DD
7. **Output Sorting**: Numeric sort by PermId

### Output Format

CSV strings with single-quoted organization names:
```
{PermId},'{Organization Name}',{Ticker},{IPO_Date}
```

**Example Output:**
```
12345,'APPLE INC.',AAPL,1980-12-12
678910,'STARBUCKS CORPORATION',SBUX,NULL
```

## Edge Cases Handled

- **Private Companies**: Ticker = NULL when `ispublic != "true"`
- **Missing IPO Dates**: IPO Date = NULL when not available
- **API Failures**: Graceful error handling with meaningful messages
- **Malformed XML**: Validation with descriptive error messages
- **Missing Fields**: NULL handling for optional data

## Error Handling

- **XML Parse Errors**: Raises `ValueError` with details
- **API Request Failures**: Raises `RuntimeError` with context
- **Missing Data**: Returns NULL for unavailable fields
- **Network Issues**: Timeout and retry capability

## Usage

```python
xml_article = """<Document>
<Title>Healthcare Disruption</Title>
<Body>Amazon, Berkshire Hathaway, and JPMorgan Chase announced...</Body>
</Document>"""

results = get_company_csv_list(xml_article)
for result in results:
    print(result)
```

## Key Features

- **Rate Limiting Compliance**: Automatic 1-second delays
- **Robust Parsing**: Handles malformed responses gracefully
- **Numeric Sorting**: PermId sorted as integers, not strings
- **Proper Quoting**: Single quotes for organization names only
- **ISO Date Conversion**: Automatic timestamp parsing
- **Modular Design**: Clean separation of concerns

## Testing

The implementation follows the HackerRank skeleton structure:
- Reads XML article from input
- Processes with `get_company_csv_list()` function
- Writes results to OUTPUT_PATH file
- Compatible with HackerRank testing environment