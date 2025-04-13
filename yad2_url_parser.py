# yad2_url_parser.py
import re
import logging
import urllib.parse

logger = logging.getLogger(__name__)

def parse_yad2_url(url):
    """פונקציה לפרסור קישור של יד2 ושליפת המזהים של היצרנים, המודלים וסוג תיבת ההילוכים
    
    Args:
        url (str): הקישור של יד2 לניתוח
        
    Returns:
        tuple: (manufacturers, models, gear_box)
        manufacturers: רשימת מזהי יצרנים או None
        models: רשימת מזהי דגמים או None
        gear_box: סוג תיבת הילוכים (101 לידני, 102 לאוטומט) או None
    """
    try:
        # פרסור הקישור
        parsed_url = urllib.parse.urlparse(url)
        query_params = urllib.parse.parse_qs(parsed_url.query)
        
        manufacturers = None
        models = None
        gear_box = None
        
        # חילוץ יצרנים
        if 'manufacturer' in query_params:
            manufacturer_str = query_params['manufacturer'][0]
            if ',' in manufacturer_str:
                manufacturers = [int(m) for m in manufacturer_str.split(',') if m.strip()]
            else:
                manufacturers = [int(manufacturer_str)]
            
        # חילוץ מודלים
        if 'model' in query_params:
            model_str = query_params['model'][0]
            if ',' in model_str:
                models = [int(m) for m in model_str.split(',') if m.strip()]
            else:
                models = [int(model_str)]
            
        # חילוץ סוג תיבת הילוכים אם קיים
        if 'gearBox' in query_params:
            gear_box_str = query_params['gearBox'][0]
            try:
                gear_box = int(gear_box_str)
            except ValueError:
                # טיפול במקרה שסוג תיבת ההילוכים אינו מספר
                pass
            
        # אם יש לנו יצרנים ומודלים, מחזירים אותם
        if manufacturers or models:
            return manufacturers, models, gear_box
            
        # התאמה לפי תבנית כגיבוי
        manufacturer_pattern = r'manufacturer=(\d+)'
        model_pattern = r'model=(\d+)'
        gear_box_pattern = r'gearBox=(\d+)'
        
        manufacturer_match = re.search(manufacturer_pattern, url)
        model_match = re.search(model_pattern, url)
        gear_box_match = re.search(gear_box_pattern, url)
        
        if manufacturer_match:
            manufacturers = [int(manufacturer_match.group(1))]
            
        if model_match:
            models = [int(model_match.group(1))]
            
        if gear_box_match:
            try:
                gear_box = int(gear_box_match.group(1))
            except ValueError:
                pass
                    
        return manufacturers, models, gear_box
            
    except Exception as e:
        logger.error(f"שגיאה בפרסור קישור יד2: {str(e)}")
        return None, None, None