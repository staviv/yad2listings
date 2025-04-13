import re
import json
import csv
from typing import List, Dict
from datetime import datetime
from bs4 import BeautifulSoup
import os
from pathlib import Path

today = datetime.now().date().strftime("%y_%m_%d")

def extract_json_from_html(html_content: str) -> Dict:
    """Extract JSON data from __NEXT_DATA__ script tag"""
    soup = BeautifulSoup(html_content, 'html.parser')
    script_tag = soup.find('script', id='__NEXT_DATA__')
    
    if script_tag is None:
        raise ValueError("Could not find __NEXT_DATA__ script tag in HTML")
        
    return json.loads(script_tag.string)

def get_month_number(month_text: str) -> int:
    # Hebrew month names to numbers mapping
    month_mapping = {
        'ינואר': 1, 'פברואר': 2, 'מרץ': 3, 'אפריל': 4,
        'מאי': 5, 'יוני': 6, 'יולי': 7, 'אוגוסט': 8,
        'ספטמבר': 9, 'אוקטובר': 10, 'נובמבר': 11, 'דצמבר': 12
    }
    return month_mapping.get(month_text, 1)  # Default to 1 if month not found

def format_date(date_str: str) -> str:
    # Parse ISO format and return YYYY-MM-DD
    return datetime.fromisoformat(date_str).strftime('%Y-%m-%d')

def calculate_years_since_production(production_year: int, production_month: int) -> float:
    production_date = datetime(production_year, production_month, 1)
    current_date = datetime.now()
    years = (current_date - production_date).days / 365.25
    return years

def process_vehicle_data(json_list: List[Dict], listing_type: str, output_file: str, mode: str = 'w') -> None:
    """Process vehicle data and write to CSV"""
    # Define the headers we want to extract
    headers = ['adNumber', 'price', 'city', 'adType', 'model', 'subModel', 
              'productionDate', 'km', 'hand', 'createdAt', 'updatedAt', 
              'rebouncedAt', 'listingType', 'number_of_years', 'km_per_year', 
              'description', 'link', 'make', 'hp', 'gearBox', 'testDate']
    
    # Open the CSV file for writing with a specific encoding
    with open(output_file, mode, newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        if mode == 'w':  # Only write header if we're creating a new file
            writer.writeheader()
        
        # Process each JSON object
        for item in json_list:
            try:
                # Create date string in YYYY-MM-DD format for production date
                year = item['vehicleDates']['yearOfProduction']
                month = get_month_number(item['vehicleDates'].get('monthOfProduction', {"text": "ינואר"})['text'])
                production_date = f"{year}-{month:02d}-01"  # Format: YYYY-MM-DD
                
                # Get test date if it exists
                test_date = ""
                if 'testDate' in item['vehicleDates'] and item['vehicleDates']['testDate']:
                    test_date = format_date(item['vehicleDates']['testDate'])
                
                # Calculate years since production
                years_since_production = calculate_years_since_production(year, month)
                
                # Calculate km per year
                km = item['km']
                km_per_year = round(km / years_since_production if years_since_production > 0 else km, 2)
                
                # Get gear type
                gear_box = ""
                if 'gearBox' in item and item['gearBox'] and 'text' in item['gearBox']:
                    gear_box = item['gearBox']['text']
                
                # Clean description text of any problematic characters
                description = item["metaData"]["description"]
                if description:
                    # Replace any problematic characters
                    description = ''.join(c if ord(c) < 128 or c.isalpha() else ' ' for c in description)
                
                # Ensure price is a number
                try:
                    price = int(item['price'])
                except (ValueError, TypeError):
                    price = 0
                
                row = {
                    'adNumber': item['adNumber'],
                    'price': price,  # Ensure this is an integer
                    'city': item['address'].get('city',{"text":""})['text'],
                    'adType': item['adType'],
                    'model': item['model']['text'],
                    'subModel': item['subModel']['text'],
                    'hp': int(re.search(r'(\d+)\s*כ״ס', item['subModel']['text']).group(1)) if re.search(r'(\d+)\s*כ״ס', item['subModel']['text']) else 0,
                    'make': item['manufacturer']['text'],
                    'productionDate': production_date,
                    'testDate': test_date,
                    'km': int(item['km']),  # Ensure this is an integer
                    'hand': int(item["hand"]["id"]),  # Ensure this is an integer
                    'gearBox': gear_box,
                    'createdAt': format_date(item['dates']['createdAt']),
                    'updatedAt': format_date(item['dates']['updatedAt']),
                    'rebouncedAt': format_date(item['dates']['rebouncedAt']),
                    'listingType': listing_type,
                    'number_of_years': float(years_since_production),  # Ensure this is a float
                    'km_per_year': float(km_per_year),  # Ensure this is a float
                    'description': description,
                    'link': f'https://www.yad2.co.il/vehicles/item/{item["token"]}',
                }
                writer.writerow(row)
            except KeyError as e:
                print(f"Skipping item due to missing key: {e}")
                print(item)
            except Exception as e:
                print(f"Error processing item: {e}")



def process_directory(directory_path: str) -> None:
    """Process all HTML files in a directory and combine the data"""
    # Get directory name for the output file
    dir_name = Path(directory_path).name
    output_file = f"{dir_name}_summary.csv"
    output_path = os.path.join(directory_path, output_file)
    
    # Process each HTML file in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith('.html') and today in filename:
            file_path = os.path.join(directory_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    print(f"Processing {filename}...")
                    html_content = file.read()
                    data = extract_json_from_html(html_content)
                    listings_data = data['props']['pageProps']['dehydratedState']['queries'][0]['state']['data']
                    
                    # Process commercial listings
                    commercial_list = listings_data.get('commercial', [])
                    if commercial_list:
                        mode = 'a' if os.path.exists(output_path) else 'w'
                        process_vehicle_data(commercial_list, 'commercial', output_path, mode)
                        print(f"Processed {len(commercial_list)} commercial listings")
                    
                    # Process private listings
                    private_list = listings_data.get('private', [])
                    if private_list:
                        mode = 'a' if os.path.exists(output_path) else 'w'
                        process_vehicle_data(private_list, 'private', output_path, mode)
                        print(f"Processed {len(private_list)} private listings")
                    
                    # Process solo listings
                    private_list = listings_data.get('solo', [])
                    if private_list:
                        mode = 'a' if os.path.exists(output_path) else 'w'
                        process_vehicle_data(private_list, 'solo', output_path, mode)
                        print(f"Processed {len(private_list)} solo listings")
                    
                    # Process platinum listings
                    private_list = listings_data.get('platinum', [])
                    if private_list:
                        mode = 'a' if os.path.exists(output_path) else 'w'
                        process_vehicle_data(private_list, 'platinum', output_path, mode)
                        print(f"Processed {len(private_list)} platinum listings")
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    print(f"Output saved to: {output_path}")
