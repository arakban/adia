Here's the complete transcription for Task 2:

```markdown
# Task 2: Tickers from Articles

## Problem Statement

You have been tasked with extracting out certain information from news articles relating to public companies as one step in part of a larger data processing pipeline. Luckily, you have been given temporary access to the Thomson Reuters' Calais web service via the Refinitiv Intelligent Tagging API to assist you with this. This web service takes an article's title and text as input, and will respond with a listing of any entities that were identified within its contents along with their Perm Ids (open, unique entity identifiers) and other relevant info.

## Assigned Task

Your goal is to extract the:
1. **organization PermId**
2. **organization Name**, and
3. **its Ticker symbol** (if available)

of the company entities identified from the article by the API service above.

In addition, to support the downstream analysis that another team will undertake, you have been asked to include:
4. **the IPO date for each public company** (if available), which you have found that you can obtain via another HTTP request outlined below.

**Rate Limiting**: Due to your completing this within the dev environment where the web service has a lower rate limit, you have been informed that **your code should wait for 1 second before each API call**.

---

## API 1: Intelligent Tags API Request

**Resource URL**: `https://api-eit.refinitiv.com/permid/calais`  
**HTTP verb**: `POST`

### Request Headers:

| Key | Description / Value |
|-----|---------------------|
| X-AG-Access-Token | **Mandatory.** Your temporary API access token. **(Value: Kf1fmqa3XaGGGsh6wMw5OPlYgsHA1FTz)** |
| Content-Type | **Optional.** The request body type. **(Default Value: text/xml)** |
| outputFormat | **Mandatory.** The response body type. **(Value: application/json)** |
| x-calais-selectiveTags | **Mandatory.** To filter result on any particular tag(s). **(Potential Values: company, person, socialtags, topic)** |

### Request Body
(also a sample of a string input that your function could receive):

```xml
<Document>

<Title>Healthcare Disruption</Title>

<Body>
Amazon, Berkshire Hathaway, and JPMorgan Chase announced that they are planning to form an independent healthcare company for their United States employees.
</Body>

</Document>
```

### Response Sample
(does not correspond to the article snippet above):

```json
{
  "doc": { ... },
  "http:\/\/d.opencalais.com\/pershash-1\/75779cc5-9c06-3c87-a1f1-8094fa045122": { ... },
  "http:\/\/d.opencalais.com\/genericHasher-1\/2d67e2d7-fdad-3e4c-a84e-825badc0aa0c": { ... },
  
  "http:\/\/d.opencalais.com\/comphash-1\/4dbfaab6-f8cb-3678-8e0e-bf467e075aa6": {
    "_typeGroup": "entities",
    "_type": "Company",
    "name": "Reuters",
    ...
    "instances": [ ... ],
    "relevance": 0,
    "confidence": { ... },
    "resolutions": [
      {
        "name": "THOMSON REUTERS CORPORATION",
        "permid": "4295861160",
        "primaryric": "TRI.TO",
        "ispublic": "true",
        "commonname": "ThomsonReuters",
        "score": 0.74383277,
        "id": "https:\/\/permid.org\/1-4295861160",
        "ticker": "TRI"
      }
    ]
  },
  ...
}
```

---

## API 2: PermId Info API Request

**Resource URL**: `https://permid.org/1-{PERM_ID}`  
**HTTP verb**: `GET`

### Request Headers:

| Key | Description / Value |
|-----|---------------------|
| X-AG-Access-Token | **Mandatory.** Your temporary API access token. **(Value: Kf1fmqa3XaGGGsh6wMw5OPlYgsHA1FTz)** |

### Query Parameters:

| Key | Description / Value |
|-----|---------------------|
| format | **Mandatory.** The response body format. **(Value: json-ld)** |

### Response Sample:

```json
{
  "@id": "https://permid.org/1-4295861160",
  "@type": "tr-org:Organization",
  "tr-common:hasPermId": "4295861160",
  "hasOrganizationPrimaryQuote": "https://permid.org/1-55838860337",
  "hasPrimaryInstrument": "https://permid.org/1-8590928696",
  ...
  "hasIPODate": "2019-12-11T05:00:00Z",
  ...
  "hasPrimaryBusinessSector": "https://permid.org/1-4294952762",
  "hasPrimaryEconomicSector": "https://permid.org/1-4294952767",
  "hasPrimaryIndustryGroup": "https://permid.org/1-4294952759",
  ...
  "hasURL": "https://www.thomsonreuters.com/",
  "vcard:organization-name": "Thomson Reuters Corp",
  "@context": { ... }
}
```

---

## Function Output Format

The output of your function should be a list of comma-separated value strings (no headers necessary), **sorted by PermId**, with the following order:

**1) Organization PermId, 2) Organization Name, 3) Ticker Symbol, 4) IPO Date.**

### Formatting Requirements:

- The organization name should be quoted using **single quotes**. None of the other values should be quoted.
- If the ticker symbol and/or IPO date are unavailable, their value should be represented with **NULL**.
- If the IPO Date is available, then it should be in the **YYYY-MM-DD** format.

### Example Output:

```
"12345,'APPLE INC.',AAPL,1980-12-12",
"678910,'STARBUCKS CORPORATION',SBUX,NULL"
```

---

## Implementation Notes

1. Parse the article XML to extract title and body
2. Call Intelligent Tags API with `x-calais-selectiveTags: company`
3. Wait 1 second (rate limiting)
4. Parse response to extract company entities with `resolutions` array
5. For each company with a PermId:
   - Extract: PermId, name, ticker (if `ispublic == "true"`)
   - Call PermId Info API to get IPO date
   - Wait 1 second (rate limiting)
   - Parse `hasIPODate` field and convert to YYYY-MM-DD format
6. Sort results by PermId (numeric sort)
7. Format as CSV strings with proper quoting

## Edge Cases to Handle

- Companies without ticker symbols (private companies)
- Companies without IPO dates
- Missing or malformed API responses
- Rate limiting (ensure 1 second wait between ALL API calls)
- Proper timestamp parsing (ISO 8601 to YYYY-MM-DD)
```

This gives you the complete Task 2 specification with all API details, requirements, and output formatting rules.