# data_processor.py
import os
import logging
import pandas as pd
from pathlib import Path
import yad2_parser
from scraper import VehicleScraper, backup_data

logger = logging.getLogger(__name__)

def scrape_data(output_dir, manufacturer, model, max_pages, gear_box=None):
    """הפעלת הסקרייפר לאיסוף נתוני רכב"""
    logger.info(f"סורק נתונים עבור manufacturer={manufacturer}, model={model}, gearBox={gear_box}...")
    
    try:
        # אתחול הסקרייפר עם הפרמטרים שסופקו
        scraper = VehicleScraper(output_dir, manufacturer, model, gear_box)
        
        # הרצת הסקרייפר עם מספר מקסימלי של עמודים
        scraper.scrape_pages(max_page=max_pages)
        
    except Exception as e:
        logger.error(f"שגיאה במהלך הסריקה: {str(e)}")
        raise
    
def process_data(output_dir):
    """עיבוד קבצי HTML שנסרקו ל-CSV"""
    logger.info("מעבד קבצי HTML שנסרקו...")
    
    try:
        # קבלת שם הספרייה לשם קובץ הפלט
        dir_name = Path(output_dir).name
        
        # עיבוד קבצי ה-HTML בספרייה
        yad2_parser.process_directory(output_dir)
        
        # בניית נתיב קובץ הפלט
        output_file = f"{dir_name}_summary.csv"
        output_path = os.path.join(output_dir, output_file)
        
        # בדיקה אם קובץ ה-CSV קיים
        if not os.path.exists(output_path):
            logger.error(f"לא נמצאו נתונים מעובדים ב-{output_path}")
            raise FileNotFoundError(f"קובץ נתונים מעובדים לא נמצא: {output_path}")
            
        return output_path
        
    except Exception as e:
        logger.error(f"שגיאה בעיבוד נתונים: {str(e)}")
        raise


def load_data(csv_path):
    """טעינה והכנת נתוני CSV לויזואליזציה"""
    try:
        logger.info(f"טוען נתונים מ-{csv_path}")
        
        # ניסיון לקידודים שונים לטיפול בבעיות קידוד אפשריות
        encodings_to_try = ['utf-8', 'cp1255', 'iso-8859-8', 'windows-1255']
        
        for encoding in encodings_to_try:
            try:
                df = pd.read_csv(csv_path, encoding=encoding)
                logger.info(f"נטען בהצלחה עם קידוד {encoding}")
                break
            except UnicodeDecodeError:
                logger.warning(f"נכשל בטעינה עם קידוד {encoding}, מנסה את הבא...")
        else:
            # אם כל הקידודים נכשלים, ננסה עם טיפול בשגיאות
            df = pd.read_csv(csv_path, encoding='utf-8', encoding_errors='replace')
            logger.warning("נטען עם החלפת תווים לא חוקיים")
        
        # המרת מחיר למספרי, כאשר שגיאות הופכות ל-NaN
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        
        # סינון מכוניות ללא מחיר או מחיר = 0
        df = df[df['price'] > 0]
        
        # וידוא שעמודות מספריות אחרות מומרות כראוי
        numeric_columns = ['km', 'hand', 'hp', 'number_of_years', 'km_per_year']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # המרת מחרוזות תאריך לאובייקטי datetime
        df['productionDate'] = pd.to_datetime(df['productionDate'], errors='coerce')
        df['testDate'] = pd.to_datetime(df['testDate'], errors='coerce')
        df['createdAt'] = pd.to_datetime(df['createdAt'], errors='coerce')
        df['updatedAt'] = pd.to_datetime(df['updatedAt'], errors='coerce')
        
        # חילוץ שנה מתאריך ייצור לסינון קל יותר
        df['productionYear'] = df['productionDate'].dt.year
        df['productionMonth'] = df['productionDate'].dt.month
        
        # עיצוב תאריכים להצגה
        df['productionDateFormatted'] = df['productionDate'].dt.strftime('%Y-%m-%d')
        df['testDateFormatted'] = df['testDate'].dt.strftime('%Y-%m-%d')
        df['createdAtFormatted'] = df['createdAt'].dt.strftime('%Y-%m-%d')
        
        # הוספת תיאור קצר (100 תווים ראשונים)
        df['shortDescription'] = df['description'].str.slice(0, 100) + '...'
        
        logger.info(f"נטענו {len(df)} רשומות רכב")
        return df
        
    except Exception as e:
        logger.error(f"שגיאה בטעינת נתונים: {str(e)}")
        raise
