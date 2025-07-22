import streamlit as st
import requests
import pandas as pd
import re
from bs4 import BeautifulSoup
from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool, AsyncOpenAI, set_tracing_disabled, RunResult
import json
from dotenv import load_dotenv
import asyncio
import os
from datetime import datetime, timedelta
import time
from streamlit_option_menu import option_menu

# --- CRITICAL: Apply nest_asyncio at the very top of the script ---
import nest_asyncio
nest_asyncio.apply()

# Load API key from .env
load_dotenv()
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
APOLLO_API_KEY = os.getenv("APOLLO_API_KEY")

st.set_page_config(page_title="Lead Scraper Pro", layout="centered")

# Create the tab navigation
selected = option_menu(
    menu_title=None,
    options=["👤 Lead Contacts", "💰 Company Funding"],
    icons=["person-lines-fill", "cash-stack"],
    default_index=0,
    orientation="horizontal",
)

# --- Initialize common components ---
provider = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    openai_client=provider,
    model="gemini-2.0-flash"
)
set_tracing_disabled(disabled=True)

@function_tool
def google_search_tool(query: str) -> list:
    """
    Performs a Google search using the Serper API and returns a list of organic URLs.
    """
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "q": query,
        "num": 100
    }
    response = requests.post("https://google.serper.dev/search", headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        if result.get("organic"):
            urls = [item.get("link") for item in result.get("organic", []) if item.get("link")]
            return urls
        else:
            return []
    else:
        print(f"Serper API Error: {response.status_code} - {response.text}")
        return []

def scrape_lead_data(url):
    """
    Scrapes a given URL for email, phone, name, company, and company URL.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            print(f"Failed to fetch {url} with status code {response.status_code}")
            return None

        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text(separator=' ', strip=True)

        # Extract emails
        emails = re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
        cleaned_emails = ", ".join(set(emails)) if emails else None

        # Extract and format phone numbers
        phones = re.findall(r"(\+?\d[\d\s().-]{6,}\d)", text)
        cleaned_phones = []
        for phone in set(phones):
            clean_num = re.sub(r"[()\s.-]", "", phone)
            if len(clean_num) >= 7 and len(clean_num) <= 15 and clean_num.lstrip('+').isdigit():
                if clean_num.startswith("00"):
                    clean_num = "+" + clean_num[2:]
                elif not clean_num.startswith('+') and not clean_num.startswith('0'):
                    clean_num = "+" + clean_num
                cleaned_phones.append(clean_num)
        formatted_phones = ", ".join(sorted(list(set(cleaned_phones)))) if cleaned_phones else None

        if not cleaned_emails and not formatted_phones:
            return None

        # Extract Name
        name = None
        match_name = re.search(r"(?:[Nn]ame|Full Name)[:\s\-]*([A-Z][a-z]+\s(?:[A-Z][a-z]+\s?)+)", text)
        if match_name:
            name = match_name.group(1).strip()
        if not name:
            meta_name = soup.find('meta', attrs={'property': 'og:title'}) or \
                        soup.find('meta', attrs={'name': 'author'})
            if meta_name and meta_name.get('content'):
                meta_content = meta_name['content']
                if re.search(r'[A-Z][a-z]+\s[A-Z][a-z]+', meta_content):
                    name_from_meta = meta_content.split('|')[0].strip()
                    name = name_from_meta.split('-')[0].strip()
                    if "profile" in name.lower() or "page" in name.lower() or "user" in name.lower():
                        name = None

        # Extract Title and Company Name
        title = soup.title.string.strip() if soup.title else ""
        possible_company = None
        for line in text.splitlines():
            if " at " in line.lower():
                parts = line.split(" at ")  
                if len(parts) > 1 and len(parts[-1].strip()) > 2:
                    possible_company = parts[-1].strip()
                    possible_company = re.sub(r'\s*(Pvt|Ltd|LLC|Inc|GmbH|Corp)\.?\s*$', '', possible_company, flags=re.IGNORECASE).strip()
                    break
        
        if not possible_company:
            domain_match = re.search(r"https?://(?:www\.)?([a-zA-Z0-9-]+\.[a-zA-Z]{2,})", url)
            if domain_match:
                domain = domain_match.group(1)
                domain_parts = domain.split('.')
                if len(domain_parts) > 1:
                    company_from_domain = domain_parts[0].replace('-', ' ').title()
                    if domain_parts[0] not in ["linkedin", "github", "facebook", "twitter", "threads"]:
                        possible_company = company_from_domain

        # Extract Company URL
        company_url = None
        for a in soup.find_all("a", href=True):
            href = a['href']
            if any(kw in href.lower() for kw in ["about", "company", "careers", "contact"]) and href.startswith("http"):
                base_domain_scraped = re.search(r"https?://(?:www\.)?([a-zA-Z0-9-]+\.[a-zA-Z]{2,})", url)
                base_domain_href = re.search(r"https?://(?:www\.)?([a-zA-Z0-9-]+\.[a-zA-Z]{2,})", href)
                
                if base_domain_scraped and base_domain_href and \
                   base_domain_scraped.group(1) == base_domain_href.group(1):
                    company_url = href
                    break
        if not company_url:
            parsed_url = requests.utils.urlparse(url)
            company_url = f"{parsed_url.scheme}://{parsed_url.netloc}"

        return {
            "URL": url,
            "Email": cleaned_emails,
            "Phone": formatted_phones,
            "Name": name,  # From your existing code
            "Company Name": possible_company or title,  # From your existing code
            "Company URL": company_url,  # From your existing code
            "Contact Type": "Email Only" if cleaned_emails and not formatted_phones 
                          else "Phone Only" if formatted_phones and not cleaned_emails
                          else "Both"
        }
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

def scrape_company_funding_data(url):
    """
    Scrapes a given URL for company funding details.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            print(f"Failed to fetch {url} with status code {response.status_code}")
            return None

        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text(separator=' ', strip=True)

        company_name = None
        funding_amount = None
        funding_round = None
        funding_date = None
        company_website = None

        # Try to extract company name from title or headings
        if soup.title:
            company_name_match = re.search(r"^(.*?)(?: Funding| raises| acquires| Series \w| Round| on Crunchbase| on TechCrunch)", soup.title.string, re.IGNORECASE)
            if company_name_match:
                company_name = company_name_match.group(1).strip()
            else:
                h1_tag = soup.find('h1')
                if h1_tag:
                    company_name = h1_tag.get_text(strip=True)

        # Extract funding amount
        amount_matches = re.findall(r"(\$|€|£)\s*([\d,]+\.?\d*)\s*(million|billion|M|B)", text, re.IGNORECASE)
        if amount_matches:
            best_match = None
            for currency, value, magnitude in amount_matches:
                amount_num = float(value.replace(',', ''))
                if 'billion' in magnitude.lower() or 'b' == magnitude.lower():
                    actual_amount = amount_num * 1_000_000_000
                elif 'million' in magnitude.lower() or 'm' == magnitude.lower():
                    actual_amount = amount_num * 1_000_000
                else:
                    actual_amount = amount_num

                if best_match is None or actual_amount > best_match[1]:
                    best_match = (f"{currency}{value} {magnitude}", actual_amount)
            if best_match:
                funding_amount = best_match[0]

        # Extract funding round
        round_matches = re.findall(r"(Seed|Series [A-Z]|Pre-seed|Angel|Venture|Growth|Debt) Round", text, re.IGNORECASE)
        if round_matches:
            funding_round = ", ".join(set(round_matches))

        # Extract funding date
        date_matches = re.findall(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2},\s+\d{4}\b|\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4}\b|\b\d{4}-\d{2}-\d{2}\b", text)
        if date_matches:
            parsed_dates = []
            for date_str in date_matches:
                try:
                    if re.match(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2},\s+\d{4}\b", date_str):
                        parsed_dates.append(datetime.strptime(date_str, "%b %d, %Y"))
                    elif re.match(r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4}\b", date_str):
                        parsed_dates.append(datetime.strptime(date_str, "%d %b %Y"))
                    elif re.match(r"\b\d{4}-\d{2}-\d{2}\b", date_str):
                        parsed_dates.append(datetime.strptime(date_str, "%Y-%m-%d"))
                except ValueError:
                    continue
            if parsed_dates:
                funding_date = max(parsed_dates).strftime("%Y-%m-%d")

        # Extract Company Website
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            if 'crunchbase.com' in url and '/organization/' in href and '/organization/' not in url and 'http' in href:
                if not re.search(r'/(person|event|fund|article|deal)/', href):
                    company_website = href
                    break
            elif any(domain in url for domain in ["techcrunch.com", "dealstreetasia.com", "venturebeat.com"]):
                if "http" in href and not any(f"/{domain_part}/" in href for domain_part in url.split('/')[2].split('.')):
                    if "company" in a_tag.get_text(strip=True).lower() or "website" in a_tag.get_text(strip=True).lower() or "homepage" in a_tag.get_text(strip=True).lower():
                        company_website = href
                        break
                    if "http" in href and not re.match(r'.*?(twitter|facebook|linkedin|instagram)\.com.*', href, re.IGNORECASE) and 'mailto:' not in href:
                        company_website = href
                        break

        if not company_name and not funding_amount:
            return None

        return {
            "URL": url,
            "Company Name": company_name,
            "Funding Amount": funding_amount,
            "Funding Round": funding_round,
            "Funding Date": funding_date,
            "Company Website": company_website,
            "Scraping Error": None
        }
    except requests.exceptions.RequestException as req_err:
        print(f"Network/Request error scraping {url}: {req_err}")
        return None
    except Exception as e:
        print(f"General error parsing content from {url}: {e}")
        return None

def get_company_and_ceo_details_from_apollo(companies_df):
    """
    Calls Apollo.io API to find comprehensive company details and CEO info.
    """
    if not APOLLO_API_KEY:
        st.warning("Apollo API Key not set. Skipping comprehensive enrichment.")
        return pd.DataFrame()

    domains_to_enrich = []
    domain_to_original_rows = {} 

    for index, row in companies_df.iterrows():
        company_website = row.get("Company Website")
        if company_website:
            domain = company_website.replace("http://", "").replace("https://", "").split("/")[0]
            if domain:
                if domain not in domains_to_enrich:
                    domains_to_enrich.append(domain)
                if domain not in domain_to_original_rows:
                    domain_to_original_rows[domain] = []
                domain_to_original_rows[domain].append(row.to_dict())

    if not domains_to_enrich:
        return pd.DataFrame()

    st.info(f"Attempting to enrich {len(domains_to_enrich)} unique company domains via Apollo.io...")

    bulk_enrich_url = "https://api.apollo.io/api/v1/organizations/bulk_enrich" 
    headers = {
        "accept": "application/json",
        "Cache-Control": "no-cache",
        "Content-Type": "application/json",
        "X-Api-Key": APOLLO_API_KEY
    }
    
    chunk_size = 50 
    domain_chunks = [domains_to_enrich[i:i + chunk_size] for i in range(0, len(domains_to_enrich), chunk_size)]

    apollo_org_data_by_domain = {} 

    for i, chunk in enumerate(domain_chunks):
        payload = {
            "domains": chunk
        }
        try:
            response = requests.post(bulk_enrich_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status() 
            bulk_data = response.json()
            
            organizations_in_response = bulk_data.get("organizations", []) 

            if organizations_in_response:
                for org in organizations_in_response:
                    org_primary_domain = org.get("primary_domain")
                    if org_primary_domain:
                        apollo_org_data_by_domain[org_primary_domain] = org
            
            st.write(f"Processed bulk enrichment chunk {i+1}/{len(domain_chunks)}. Found {len(organizations_in_response)} orgs.")

        except requests.exceptions.RequestException as e:
            st.error(f"Apollo Bulk Enrichment API error for chunk {i+1}: {e}. Check API key and endpoint.")
        except json.JSONDecodeError:
            st.error(f"Apollo Bulk Enrichment API: Failed to parse JSON response for chunk {i+1}. Response: {response.text}")
        except Exception as e:
            st.error(f"Error processing Apollo bulk data for chunk {i+1}: {e}")
        
        time.sleep(0.5) 

    final_enriched_rows = []

    for original_row_dict in companies_df.to_dict(orient='records'):
        current_domain = original_row_dict.get("Company Website", "").replace("http://", "").replace("https://", "").split("/")[0]
        
        apollo_company_phone = None
        apollo_employees = None
        apollo_industry = None
        apollo_location = None
        apollo_revenue = None
        ceo_name = None
        ceo_email = None
        ceo_phone = None 

        org_data = apollo_org_data_by_domain.get(current_domain)
        
        if org_data:
            org_id = org_data.get("id")
            apollo_company_phone = org_data.get("phone") 
            apollo_employees = org_data.get("estimated_num_employees")
            apollo_industry = org_data.get("industry")
            
            apollo_location_parts = []
            if org_data.get("city"): apollo_location_parts.append(org_data["city"])
            if org_data.get("state"): apollo_location_parts.append(org_data["state"])
            if org_data.get("country"): apollo_location_parts.append(org_data["country"])
            apollo_location = ", ".join(apollo_location_parts) if apollo_location_parts else None
            
            revenue_range = org_data.get("revenue_range")
            if revenue_range:
                min_rev = revenue_range.get('min')
                max_rev = revenue_range.get('max')
                if min_rev is not None and max_rev is not None:
                    apollo_revenue = f"${min_rev:,}-${max_rev:,}"
                elif min_rev is not None:
                    apollo_revenue = f"${min_rev:,}+"
            
            if org_id:
                people_search_url = "https://api.apollo.io/api/v1/people/search" 
                people_params = {
                    "organization_ids[]": [org_id],
                    "q_job_titles": "CEO,Chief Executive Officer,Founder,Co-founder", 
                    "per_page": 1,
                    "sort_by_field": "likelihood", 
                    "sort_ascending": False,
                    "person_locations": [apollo_location] if apollo_location else [] 
                }
                try:
                    person_response = requests.get(people_search_url, headers=headers, params=people_params, timeout=10)
                    person_response.raise_for_status()
                    person_data = person_response.json()

                    if person_data.get("people") and len(person_data["people"]) > 0:
                        ceo = person_data["people"][0]
                        ceo_name = f"{ceo.get('first_name', '')} {ceo.get('last_name', '')}".strip()
                        ceo_email = ceo.get("email")
                        if ceo.get("phone_numbers"):
                            for phone_obj in ceo["phone_numbers"]:
                                if phone_obj.get("sanitized_number"):
                                    ceo_phone = phone_obj["sanitized_number"]
                                    break 
                        
                except requests.exceptions.RequestException as e:
                    print(f"Apollo People Search API error for Org ID {org_id}: {e}")
                except json.JSONDecodeError:
                    st.error(f"Apollo People Search API: Failed to parse JSON response for Org ID {org_id}. Response: {person_response.text}")
                except Exception as e:
                    st.error(f"Error processing Apollo people data for Org ID {org_id}: {e}")
        
        row_with_apollo_data = original_row_dict.copy()
        row_with_apollo_data["Company Phone (Apollo)"] = apollo_company_phone
        row_with_apollo_data["Employees (Apollo)"] = apollo_employees
        row_with_apollo_data["Industry (Apollo)"] = apollo_industry
        row_with_apollo_data["Location (Apollo)"] = apollo_location
        row_with_apollo_data["Revenue (Apollo)"] = apollo_revenue
        row_with_apollo_data["CEO Name (Apollo)"] = ceo_name
        row_with_apollo_data["CEO Email (Apollo)"] = ceo_email
        row_with_apollo_data["CEO Phone (Apollo)"] = ceo_phone 

        final_enriched_rows.append(row_with_apollo_data)

    return pd.DataFrame(final_enriched_rows)

# Define the Agent
agent = Agent(
    name="Search Agent",
    instructions="You are a lead generation bot. Use the google_search_tool to find URLs based on the query. Return the complete list of all URLs extracted (up to 100), exactly as provided by the tool, with no filtering, truncation, or additional text.",
    tools=[google_search_tool],
    model=model
)

# --- Lead Contacts Tab ---
if selected == "👤 Lead Contacts":
    st.header("👤 Lead Contact Scraper")
    st.markdown("Find emails, phone numbers, and other contact details from public profiles.")
    
    col1, col2 = st.columns(2)
    with col1:
        job_title = st.text_input("Job Title", "Data Scientist", key="job_title_lead")
    with col2:
        location = st.text_input("Location", "San Francisco", key="location_lead")
    
    industry = st.text_input("Industry (Optional)", "", key="industry_lead")
    start_lead_scraping = st.button("🔎 Start Lead Contact Scraping", key="lead_button")

    if start_lead_scraping:
        all_found_urls = []
        flexible_job_query = job_title.strip()
        flexible_loc_query = location.strip()
        flexible_industry_query = industry.strip()

        TARGET_SITES_LEADS = ["linkedin.com", "github.com", "facebook.com", "twitter.com", "threads.net"]
        
        st.subheader("Collecting URLs for Individual Leads...")
        for site_index, site in enumerate(TARGET_SITES_LEADS):
            with st.spinner(f"🤖 Running AI Agent to search leads on {site} ({site_index + 1}/{len(TARGET_SITES_LEADS)})..."):
                try:
                    search_query_parts = [
                        f"site:{site}",
                    ]
                    
                    if flexible_job_query:
                        search_query_parts.append(f'"{flexible_job_query}"')
                    if flexible_loc_query:
                        search_query_parts.append(f'"{flexible_loc_query}"')
                    if flexible_industry_query:
                        search_query_parts.append(f'"{flexible_industry_query}"')

                    final_search_query = " ".join(filter(None, search_query_parts))

                    prompt = f"""
                    Search Google with this query:
                    {final_search_query}
                    Return only a clean list of 100 working URLs from public pages. No need to explain anything.
                    """
                    
                    result = Runner.run_sync(starting_agent=agent, input=prompt)
                    
                    if isinstance(result, RunResult):
                        output_str = result.final_output
                        urls = []
                        if output_str:
                            try:
                                parsed_list = json.loads(output_str)
                                if isinstance(parsed_list, list):
                                    urls = [item for item in parsed_list if item.startswith('http')]
                            except (json.JSONDecodeError, TypeError):
                                lines = output_str.replace(', ', '\n').split('\n')
                                urls = [line.strip() for line in lines if line.strip().startswith('http')]
                        
                        all_found_urls.extend(urls)
                    else:
                        st.warning(f"⚠️ Unexpected result type for {site}: {type(result)}. Skipping this site.")
                        continue

                    if not urls:
                        st.warning(f"⚠️ No URLs returned from the search for {site} with the given criteria.")
                    else:
                        st.success(f"✅ Fetched {len(urls)} URLs from {site}. Continuing search...")

                except Exception as e:
                    st.error(f"🚫 Error searching {site}: {str(e)}")

        if not all_found_urls:
            st.warning("⚠️ No URLs found across all searches for individual leads. Try broadening your search terms or checking inputs.")
        else:
            unique_urls_to_scrape = list(set(all_found_urls))
            st.success(f"✅ Total {len(unique_urls_to_scrape)} unique URLs to scrape for individual leads. Extracting contact data now...")
            lead_data = []

            progress_text = "Processing URLs for individual lead data extraction, please wait..."
            my_bar = st.progress(0, text=progress_text)
            total_urls_to_process = len(unique_urls_to_scrape)

            for i, url in enumerate(unique_urls_to_scrape):
                data = scrape_lead_data(url)
                if data:
                    lead_data.append(data)
                
                progress = (i + 1) / total_urls_to_process
                my_bar.progress(progress, text=f"{progress_text} ({i+1}/{total_urls_to_process})")

            my_bar.empty()

            # After scraping completes
            if lead_data:
                df = pd.DataFrame(lead_data)
                
                # New: Categorize leads
                df['Contact Type'] = df.apply(
                    lambda x: "Email Only" if x['Email'] and not x['Phone'] else
                            "Phone Only" if x['Phone'] and not x['Email'] else
                            "Both",
                    axis=1
                )
                
                # Display stats
                st.write(f"Total Leads Found: {len(df)}")
                st.write(f"Email Only: {len(df[df['Contact Type']=='Email Only'])}")
                st.write(f"Phone Only: {len(df[df['Contact Type']=='Phone Only'])}")
                st.write(f"Both Email & Phone: {len(df[df['Contact Type']=='Both'])}")
                
                # Filter options
                view_option = st.radio(
                    "Show:",
                    ["All Leads", "Email Only", "Phone Only", "Both Email & Phone"]
                )
                
                if view_option == "Email Only":
                    df = df[df['Contact Type'] == "Email Only"]
                elif view_option == "Phone Only":
                    df = df[df['Contact Type'] == "Phone Only"]
                elif view_option == "Both Email & Phone":
                    df = df[df['Contact Type'] == "Both"]
                
                # Display the filtered dataframe
                st.dataframe(df)
                
                # Download options
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"Download {view_option} Leads",
                    data=csv,
                    file_name=f"{view_option.lower().replace(' ', '_')}_leads.csv",
                    mime="text/csv"
                )

# --- Company Funding Tab ---
elif selected == "💰 Company Funding":
    st.header("💰 Company Funding Scraper")
    st.markdown("Find recently funded companies and their details.")
    
    col1, col2 = st.columns(2)
    with col1:
        funding_keywords = st.text_input("Keywords (e.g., AI, Fintech, SaaS)", "AI", key="funding_keywords")
    with col2:
        default_date = datetime.now() - timedelta(days=30)
        funding_start_date = st.date_input("Funding Activity After Date", default_date, key="funding_date")
    
    start_company_scraping = st.button("🔎 Start Company Funding Scraping", key="funding_button")

    if start_company_scraping:
        all_funding_urls = []
        formatted_date = funding_start_date.strftime("%Y-%m-%d")

        TARGET_SITES_FUNDING = ["crunchbase.com", "techcrunch.com", "dealstreetasia.com", "venturebeat.com"]

        st.subheader("Collecting URLs for Company Funding Information...")
        for site_index, site in enumerate(TARGET_SITES_FUNDING):
            with st.spinner(f"🤖 Running AI Agent to search funding info on {site} ({site_index + 1}/{len(TARGET_SITES_FUNDING)})..."):
                try:
                    search_query_parts = [
                        f"site:{site}",
                        f"intitle:\"raises\"",
                        f"(\"million\" OR \"billion\")",
                        f"after:{formatted_date}",
                    ]
                    if funding_keywords.strip():
                        search_query_parts.append(f'"{funding_keywords.strip()}"')

                    final_search_query = " ".join(filter(None, search_query_parts))

                    prompt = f"""
                    Search Google with this query:
                    {final_search_query}
                    Return only a clean list of 100 working URLs from public pages. No need to explain anything.
                    """
                    
                    result = Runner.run_sync(starting_agent=agent, input=prompt)
                    
                    if isinstance(result, RunResult):
                        output_str = result.final_output
                        urls = []
                        if output_str:
                            try:
                                parsed_list = json.loads(output_str)
                                if isinstance(parsed_list, list):
                                    urls = [item for item in parsed_list if item.startswith('http')]
                            except (json.JSONDecodeError, TypeError):
                                lines = output_str.replace(', ', '\n').split('\n')
                                urls = [line.strip() for line in lines if line.strip().startswith('http')]
                        
                        all_funding_urls.extend(urls)
                    else:
                        st.warning(f"⚠️ Unexpected result type for {site}: {type(result)}. Skipping this site.")
                        continue

                    if not urls:
                        st.warning(f"⚠️ No URLs returned from the search for {site} with the given criteria.")
                    else:
                        st.success(f"✅ Fetched {len(urls)} URLs from {site}. Continuing search...")

                except Exception as e:
                    st.error(f"🚫 Error searching {site}: {str(e)}")

        if not all_funding_urls:
            st.warning("⚠️ No URLs found across all searches for company funding. Try broadening your search terms or adjusting the date.")
        else:
            unique_funding_urls_to_scrape = list(set(all_funding_urls))
            st.success(f"✅ Total {len(unique_funding_urls_to_scrape)} unique URLs to scrape for company funding. Extracting data now...")
            company_funding_data = []

            progress_text = "Processing URLs for company funding data extraction, please wait..."
            my_bar = st.progress(0, text=progress_text)
            total_urls_to_process = len(unique_funding_urls_to_scrape)

            for i, url in enumerate(unique_funding_urls_to_scrape):
                data = scrape_company_funding_data(url)
                if data:
                    company_funding_data.append(data)
                
                progress = (i + 1) / total_urls_to_process
                my_bar.progress(progress, text=f"{progress_text} ({i+1}/{total_urls_to_process})")

            my_bar.empty()

            if company_funding_data:
                df_funding = pd.DataFrame(company_funding_data)
                df_funding.drop_duplicates(subset=['URL'], inplace=True)
                df_funding.drop_duplicates(subset=['Company Name', 'Funding Amount'], inplace=True, keep='first') 

                st.subheader(f"Found Company Funding Details ({len(df_funding)} Entries)")
                st.dataframe(df_funding)
                
                if not df_funding.empty:
                    st.subheader("Enriching company data with Apollo.io...")
                    df_enriched_funding = get_company_and_ceo_details_from_apollo(df_funding.copy()) 

                    if not df_enriched_funding.empty:
                        st.subheader("Enriched Company Funding Details with Comprehensive Apollo Data")
                        st.dataframe(df_enriched_funding)
                        
                        csv_enriched = df_enriched_funding.to_csv(index=False).encode('utf-8')
                        st.download_button("⬇️ Download Enriched Company Funding CSV", data=csv_enriched, file_name="enriched_company_funding_data.csv", mime="text/csv")
                    else:
                        st.info("No companies found to enrich, or Apollo API did not return data.")
                else:
                    st.info("No company funding details could be extracted from the scanned pages.")
            else:
                st.warning("⚠️ No company funding details could be extracted from the scanned pages.")
