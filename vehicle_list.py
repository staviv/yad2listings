# vehicle_list.py
import json
import os
import logging

logger = logging.getLogger(__name__)

def load_vehicle_list(file_path="vehicle_list.json"):
    """טעינת רשימת מכוניות מקובץ JSON או יצירת קובץ חדש עם מכוניות ברירת מחדל"""
    # מכוניות ברירת מחדל אם הקובץ לא קיים
    default_vehicles = [
        {"manufacturer": 38, "model": 10514, "description": "Default vehicle"}
    ]
    
    # יצירת קובץ עם מכוניות ברירת מחדל אם לא קיים
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(default_vehicles, f, indent=4)
        return default_vehicles
    
    # טעינת מכוניות מהקובץ
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            vehicles = json.load(f)
        return vehicles
    except Exception as e:
        logger.error(f"שגיאה בטעינת רשימת מכוניות: {str(e)}")
        return default_vehicles

def add_vehicle_to_list(manufacturer, model, description="", gearBox=None, file_path="vehicle_list.json"):
    """הוספת מכונית לרשימת המכוניות לסריקה"""
    # טעינת מכוניות קיימות
    vehicles = load_vehicle_list(file_path)
    
    # בדיקה אם המכונית כבר קיימת ועדכון תיבת ההילוכים אם צריך
    for vehicle in vehicles:
        if vehicle.get("manufacturer") == manufacturer and vehicle.get("model") == model:
            # בדיקה אם צריך לעדכן את תיבת ההילוכים
            if gearBox is not None and vehicle.get("gearBox") != gearBox:
                vehicle["gearBox"] = gearBox
                # שמירת הרשימה המעודכנת
                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(vehicles, f, indent=4)
                except Exception as e:
                    logger.error(f"שגיאה בשמירת רשימת המכוניות: {str(e)}")
            return vehicles  # המכונית כבר קיימת
    
    # הוספת מכונית חדשה
    new_vehicle = {
        "description": description
    }
    
    # הוספת יצרן רק אם הוא סופק
    if manufacturer is not None:
        new_vehicle["manufacturer"] = manufacturer
        
    # הוספת מודל רק אם הוא סופק
    if model is not None:
        new_vehicle["model"] = model
    
    # הוספת תיבת הילוכים אם סופקה
    if gearBox is not None:
        new_vehicle["gearBox"] = gearBox
    
    vehicles.append(new_vehicle)
    
    # שמירת המכוניות לקובץ
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(vehicles, f, indent=4)
    except Exception as e:
        logger.error(f"שגיאה בשמירת רשימת המכוניות: {str(e)}")
    
    return vehicles