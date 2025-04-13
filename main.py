# main.py
import os
import sys
import argparse
import logging
from pathlib import Path
import traceback

# יבוא מודולים אחרים של האפליקציה
from scraper import VehicleScraper, backup_data
from data_processor import scrape_data, process_data, load_data
from dashboard_app import create_dashboard
from vehicle_list import load_vehicle_list

# הגדרת לוגר
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('main')

def parse_arguments():
    """ניתוח ארגומנטים משורת הפקודה"""
    parser = argparse.ArgumentParser(description='מנתח מחירי רכב')
    parser.add_argument('--output-dir', type=str, default='scraped_vehicles',
                        help='ספריית היעד לשמירת נתונים')
    parser.add_argument('--manufacturer', type=int, default=38,
                        help='מזהה היצרן לסריקה')
    parser.add_argument('--model', type=int, default=10514,
                        help='מזהה הדגם לסריקה')
    parser.add_argument('--max-pages', type=int, default=25,
                        help='מספר מקסימלי של עמודים לסריקה')
    parser.add_argument('--skip-scrape', action='store_true',
                        help='דלג על סריקה והשתמש בנתונים קיימים')
    parser.add_argument('--port', type=int, default=8050,
                        help='פורט להפעלת שרת האינטרנט')
    return parser.parse_args()

def main():
    """פונקציה ראשית להפעלת מנתח מחירי הרכב"""
    try:
        # ניתוח ארגומנטים משורת הפקודה
        args = parse_arguments()
        
        # יצירת ספריית יעד אם לא קיימת
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # גיבוי נתונים קיימים
        backup_folder = backup_data(args.output_dir)
        logger.info(f"גובו נתונים קיימים ל-{backup_folder}")
        
        # שלב 1: סריקת הנתונים אם לא דולגו
        if not args.skip_scrape:
            # טעינת רשימת הרכבים
            vehicles = load_vehicle_list()
            
            for vehicle in vehicles:
                manufacturer = vehicle.get("manufacturer", args.manufacturer)
                model = vehicle.get("model", args.model)
                gear_box = vehicle.get("gearBox", None)
                
                logger.info(f"סורק נתונים עבור manufacturer={manufacturer}, model={model}, gearBox={gear_box}...")
                scrape_data(args.output_dir, manufacturer, model, args.max_pages, gear_box)
        
        # שלב 2: עיבוד הנתונים שנסרקו
        csv_path = process_data(args.output_dir)
        
        # שלב 3: טעינת הנתונים
        df = load_data(csv_path)
        
        # ניקוי קובץ ה-CSV לאחר הטעינה
        try:
            os.unlink(csv_path)
            logger.info(f"נמחק קובץ CSV זמני: {csv_path}")
        except Exception as e:
            logger.warning(f"לא ניתן למחוק קובץ CSV זמני: {str(e)}")
        
        # שלב 4: יצירה והפעלה של לוח המחוונים
        create_dashboard(df, args.output_dir, args, args.port)
        
    except Exception as e:
        logger.error(f"אירעה שגיאה: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
