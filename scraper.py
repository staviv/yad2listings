import requests
import time
import os
from datetime import datetime, timedelta
from pathlib import Path
import logging
import yad2_parser

class VehicleScraper:
    def __init__(self, output_dir, manufacturer=None, model=None, gearBox=None):
        """
        אתחול הסקרייפר עם ספריית פלט ופרמטרים של הרכב
        
        Args:
            output_dir (str): ספריית היעד לשמירת הקבצים
            manufacturer (int, optional): מזהה היצרן או None עבור כל היצרנים
            model (int, optional): מזהה המודל או None עבור כל המודלים
            gearBox (int, optional): 101 לידני, 102 לאוטומט, None לכל סוגי תיבות ההילוכים
        """
        self.output_dir = Path(output_dir)
        self.manufacturer = manufacturer
        self.model = model
        self.gearBox = gearBox
        self.session = requests.Session()
        
        # הגדרת כותרות הבקשה בדיוק כמו בפקודת curl
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
        
        # הגדרת עוגיות
        self.cookies = {
            '__ssds': '3',
            'y2018-2-cohort': '88',
            'use_elastic_search': '1',
            'abTestKey': '2',
            'cohortGroup': 'D'
            # הערה: נוספו רק עוגיות חיוניות. אפשר להוסיף עוד במידת הצורך.
        }
        
        # יצירת ספריית פלט אם לא קיימת
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # הגדרת לוגר
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def build_url(self, page_num):
        """בניית URL עבור מספר עמוד מסוים עם פילטרים אופציונליים"""
        base_url = "https://www.yad2.co.il/vehicles/cars"
        params = {
            # הוספת פרמטרים רק אם הם סופקו
            "page": page_num
        }
        
        # הוספת פילטר יצרן אם צוין
        if self.manufacturer is not None:
            params["manufacturer"] = self.manufacturer
            
        # הוספת פילטר מודל אם צוין
        if self.model is not None:
            params["model"] = self.model
            
        # הוספת פילטר תיבת הילוכים אם צוין
        if self.gearBox is not None:
            params["gearBox"] = self.gearBox
            
        return f"{base_url}?{'&'.join(f'{k}={v}' for k, v in params.items())}"

    def get_output_filename(self, page_num):
        """יצירת שם קובץ פלט בהתבסס על יצרן, מודל ותיבת הילוכים"""
        today = datetime.now().date().strftime("%y_%m_%d")
        filename_parts = [today]
        
        if self.manufacturer is not None:
            filename_parts.append(f"manufacturer{self.manufacturer}")
            
        if self.model is not None:
            filename_parts.append(f"model{self.model}")
            
        if self.gearBox is not None:
            filename_parts.append(f"gearBox{self.gearBox}")
            
        filename_parts.append(f"page{page_num}")
        
        return self.output_dir / f"{'_'.join(filename_parts)}.html"

    def should_skip_file(self, filepath):
        """בדיקה אם הקובץ קיים ושונה ב-24 השעות האחרונות"""
        if not filepath.exists():
            return False
            
        file_mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
        return datetime.now() - file_mtime < timedelta(days=1)

    def fetch_page(self, page_num):
        """
        הבאת עמוד בודד ושמירתו לקובץ
        
        Args:
            page_num (int): מספר העמוד להבאה
            
        Returns:
            bool: True אם העמוד הובא בהצלחה, False אם דולג או נכשל
        """
        output_file = self.get_output_filename(page_num)
        
        if self.should_skip_file(output_file):
            self.logger.info(f"דילוג על עמוד {page_num} - קיים קובץ עדכני")
            with open(output_file, 'r', encoding='utf-8') as file:
                print(f"מעבד {output_file}...")
                html_content = file.read()
                data = yad2_parser.extract_json_from_html(html_content)
                listings_data = data['props']['pageProps']['dehydratedState']['queries'][0]['state']['data']
                return listings_data["pagination"]["pages"]
            
        try:
            url = self.build_url(page_num)
            self.logger.info(f"מביא עמוד {page_num}")
            
            time.sleep(5)  # הגבלת קצב
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
                
            self.logger.info(f"נשמר בהצלחה עמוד {page_num}")
            return listings_data["pagination"]["pages"]
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"שגיאה בהבאת עמוד {page_num}: {str(e)}")
            return

    def scrape_pages(self, max_page=100):
        """
        הבאת מספר עמודים עם הגבלת קצב
        
        Args:
            max_page (int): מספר מקסימלי של עמודים להבאה
        """
        page = 1
        while True:
            pages = self.fetch_page(page)
            print (f"עמוד {page}/{pages}")
            # המתנה בין בקשות רק אם באמת ביצענו בקשה
            if pages and page < pages and page < max_page:
                page += 1
            else:
                return

def backup_data(output_dir):
    """גיבוי קבצי נתונים קיימים לספריית גיבוי"""
    import shutil
    from datetime import datetime
    import os
    import logging
    
    logger = logging.getLogger(__name__)
    
    backup_dir = os.path.join(output_dir, "backup")
    os.makedirs(backup_dir, exist_ok=True)
    
    # קבלת חותמת זמן לתיקיית הגיבוי
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_folder = os.path.join(backup_dir, timestamp)
    os.makedirs(backup_folder, exist_ok=True)
    
    # העתקת כל קבצי ה-HTML לתיקיית הגיבוי
    html_files = [f for f in os.listdir(output_dir) if f.endswith('.html')]
    if html_files:
        for html_file in html_files:
            source_path = os.path.join(output_dir, html_file)
            dest_path = os.path.join(backup_folder, html_file)
            shutil.copy2(source_path, dest_path)
        
        logger.info(f"גובו {len(html_files)} קבצי HTML ל-{backup_folder}")
    else:
        logger.info("אין קבצי HTML לגיבוי")
    
    return backup_folder

def main():
    # דוגמת שימוש
    output_dir = "scraped_vehicles"  # החלף בספריית היעד הרצויה
    VehicleScraper(output_dir, manufacturer=41, model=10574).scrape_pages(max_page=10) # Tiguan

if __name__ == "__main__":
    main()