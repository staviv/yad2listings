import requests
import time
import os
from datetime import datetime, timedelta
from pathlib import Path
import logging
import yad2_parser

class VehicleScraper:
    def __init__(self, output_dir, manufacturer=38, model=10514):
        """
        Initialize the scraper with output directory and vehicle parameters
        
        Args:
            output_dir (str): Directory to save the scraped files
            manufacturer (int): Manufacturer ID
            model (int): Model ID
        """
        self.output_dir = Path(output_dir)
        self.manufacturer = manufacturer
        self.model = model
        self.session = requests.Session()
        
        # Set up headers exactly as in the curl command
        self.headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'en-US,en;q=0.9,he;q=0.8',
            'Cache-Control': 'max-age=0',
            'Connection': 'keep-alive',
            'DNT': '1',
            'Referer': 'https://www.yad2.co.il/',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'same-origin',
            'Sec-Fetch-User': '?1',
            'Upgrade-Insecure-Requests': '1',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
            'sec-ch-ua': '"Chromium";v="131", "Not_A Brand";v="24"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"'
        }
        
        # Set up cookies
        self.cookies = {
            '__ssds': '3',
            'y2018-2-cohort': '88',
            'use_elastic_search': '1',
            'abTestKey': '2',
            'cohortGroup': 'D'
            # Note: Added only essential cookies. Add more if needed.
        }
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def build_url(self, page_num):
        """Build the URL for a specific page number"""
        base_url = "https://www.yad2.co.il/vehicles/cars"
        params = {
            "manufacturer": self.manufacturer,
            "model": self.model,
            # No hand filter to get all vehicles
            "page": page_num
        }
        return f"{base_url}?{'&'.join(f'{k}={v}' for k, v in params.items())}"

    def get_output_filename(self, page_num):
        today = datetime.now().date().strftime("%y_%m_%d")
        """Generate output filename based on manufacturer and model"""
        return self.output_dir / f"{today}_manufacturer{self.manufacturer}_model{self.model}_page{page_num}.html"

    def should_skip_file(self, filepath):
        """Check if file exists and was modified in the last 24 hours"""
        if not filepath.exists():
            return False
            
        file_mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
        return datetime.now() - file_mtime < timedelta(days=1)

    def fetch_page(self, page_num):
        """
        Fetch a single page and save it to file
        
        Args:
            page_num (int): Page number to fetch
            
        Returns:
            bool: True if page was fetched successfully, False if skipped or failed
        """
        output_file = self.get_output_filename(page_num)
        
        if self.should_skip_file(output_file):
            self.logger.info(f"Skipping page {page_num} - recent file exists")
            with open(output_file, 'r', encoding='utf-8') as file:
                print(f"Processing {output_file}...")
                html_content = file.read()
                data = yad2_parser.extract_json_from_html(html_content)
                listings_data = data['props']['pageProps']['dehydratedState']['queries'][0]['state']['data']
                return listings_data["pagination"]["pages"]
            
        try:
            url = self.build_url(page_num)
            self.logger.info(f"Fetching page {page_num}")
            
            time.sleep(5)  # Rate limiting
            response = self.session.get(
                url,
                headers=self.headers,
                cookies=self.cookies,
                allow_redirects=True
            )
            response.raise_for_status()

            assert len(response.content) > 50000 and b'__NEXT_DATA__' in response.content, len(response.content)
             
            data = yad2_parser.extract_json_from_html(response.content.decode("utf-8"))
            listings_data = data['props']['pageProps']['dehydratedState']['queries'][0]['state']['data']
            with open(output_file, 'wb') as f:
                f.write(response.content)
                
            self.logger.info(f"Successfully saved page {page_num}")
            return listings_data["pagination"]["pages"]
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching page {page_num}: {str(e)}")
            return

    def scrape_pages(self, max_page=100):
        """
        Fetch multiple pages with rate limiting
        
        Args:
            num_pages (int): Number of pages to fetch
        """
        page = 1
        while True:
            pages = self.fetch_page(page)
            print (f"Page {page}/{pages}")
            # Only wait between requests if we actually made a request
            if pages and page < pages and page < max_page:
                page += 1
            else:
                return

def main():
    # Example usage
    output_dir = "scraped_vehicles"  # Replace with your desired output directory
    # VehicleScraper(output_dir, manufacturer=32, model=1337).scrape_pages() # Nissan
    # return
    # VehicleScraper(output_dir, manufacturer=19, model=12894).scrape_pages(max_page=20) # bz4x
    # VehicleScraper(output_dir, manufacturer=32, model=10449).scrape_pages(max_page=20) # Nissan
    # VehicleScraper(output_dir, manufacturer=21, model=10283).scrape_pages(max_page=1) 
    # VehicleScraper(output_dir, manufacturer=41, model=11579).scrape_pages(max_page=5) # ID.4
    # VehicleScraper(output_dir, manufacturer=41, model=12928).scrape_pages(max_page=5) # ID.5
    # VehicleScraper(output_dir, manufacturer=40, model=10545).scrape_pages(max_page=5)
    # VehicleScraper(output_dir, manufacturer=21, model=11239).scrape_pages(max_page=10) # Ioniq 5
    # VehicleScraper(output_dir, manufacturer=92, model=12134).scrape_pages(max_page=10) # Cupra Formentor
    # VehicleScraper(output_dir, manufacturer=41, model=10574).scrape_pages(max_page=10) # Tiguan
    # VehicleScraper(output_dir, manufacturer=40, model=11568).scrape_pages(max_page=10) # Enyaq
    # manufacturer=19&model=10226&subModel=104254,104255,104253
    # VehicleScraper(output_dir, manufacturer=19, model=10226).scrape_pages(max_page=20)
    # scraper = VehicleScraper(output_dir, manufacturer=21, model=10283)
    # scraper = VehicleScraper(output_dir, manufacturer=40, model=10545)
    # scraper = VehicleScraper(output_dir, manufacturer=21, model=11239)
    # scraper = VehicleScraper(output_dir, manufacturer=41, model=10574)

if __name__ == "__main__":
    main()