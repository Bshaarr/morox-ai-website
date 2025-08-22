from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
import requests
from io import BytesIO
import base64
import os
from dotenv import load_dotenv

# تحميل المتغيرات البيئية
load_dotenv()

app = Flask(__name__)
CORS(app)

# تهيئة نموذج وصف الصور
def load_image_captioning_model():
    """تحميل نموذج وصف الصور"""
    try:
        # استخدام نموذج متعدد اللغات لوصف الصور
        model_name = "microsoft/git-base-coco"
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForVision2Seq.from_pretrained(model_name)
        return processor, model
    except Exception as e:
        print(f"خطأ في تحميل النموذج: {e}")
        return None, None

# تحميل النموذج عند بدء التطبيق
processor, model = load_image_captioning_model()

def describe_image_english(image):
    """وصف الصورة باللغة الإنجليزية"""
    try:
        if processor is None or model is None:
            return "Model not loaded"
        
        # معالجة الصورة
        inputs = processor(images=image, return_tensors="pt")
        
        # توليد الوصف
        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values=inputs.pixel_values,
                max_length=50,
                num_beams=4,
                early_stopping=True
            )
        
        # تحويل المعرفات إلى نص
        description = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return description.strip()
    except Exception as e:
        return f"Error generating English description: {str(e)}"

def describe_image_arabic(image):
    """وصف الصورة باللغة العربية"""
    try:
        if processor is None or model is None:
            return "النموذج غير محمل"
        
        # معالجة الصورة
        inputs = processor(images=image, return_tensors="pt")
        
        # توليد الوصف باللغة العربية
        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values=inputs.pixel_values,
                max_length=50,
                num_beams=4,
                early_stopping=True,
                do_sample=True,
                temperature=0.7
            )
        
        # تحويل المعرفات إلى نص
        description = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # ترجمة بسيطة للكلمات الأساسية (يمكن تحسينها)
        arabic_translations = {
            "a person": "شخص",
            "a man": "رجل",
            "a woman": "امرأة",
            "a child": "طفل",
            "a dog": "كلب",
            "a cat": "قط",
            "a car": "سيارة",
            "a building": "مبنى",
            "a tree": "شجرة",
            "a flower": "زهرة",
            "a table": "طاولة",
            "a chair": "كرسي",
            "a book": "كتاب",
            "a phone": "هاتف",
            "a computer": "حاسوب",
            "a camera": "كاميرا",
            "a street": "شارع",
            "a road": "طريق",
            "a mountain": "جبل",
            "a sea": "بحر",
            "a river": "نهر",
            "a sky": "سماء",
            "a sun": "شمس",
            "a moon": "قمر",
            "a star": "نجمة",
            "a cloud": "سحابة",
            "a rain": "مطر",
            "a snow": "ثلج",
            "a fire": "نار",
            "a water": "ماء",
            "a food": "طعام",
            "a drink": "شراب",
            "a shirt": "قميص",
            "a pants": "بنطلون",
            "a hat": "قبعة",
            "a shoe": "حذاء",
            "a bag": "حقيبة",
            "a clock": "ساعة",
            "a door": "باب",
            "a window": "نافذة",
            "a wall": "جدار",
            "a floor": "أرضية",
            "a ceiling": "سقف",
            "a light": "ضوء",
            "a shadow": "ظل",
            "a color": "لون",
            "a red": "أحمر",
            "a blue": "أزرق",
            "a green": "أخضر",
            "a yellow": "أصفر",
            "a black": "أسود",
            "a white": "أبيض",
            "a big": "كبير",
            "a small": "صغير",
            "a tall": "طويل",
            "a short": "قصير",
            "a beautiful": "جميل",
            "a nice": "جميل",
            "a good": "جيد",
            "a bad": "سيء",
            "a happy": "سعيد",
            "a sad": "حزين",
            "a young": "شاب",
            "a old": "عجوز",
            "a new": "جديد",
            "a old": "قديم"
        }
        
        # تطبيق الترجمات
        arabic_description = description
        for english, arabic in arabic_translations.items():
            arabic_description = arabic_description.replace(english, arabic)
        
        return arabic_description.strip()
    except Exception as e:
        return f"خطأ في توليد الوصف العربي: {str(e)}"

@app.route('/')
def home():
    """الصفحة الرئيسية"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/describe', methods=['POST'])
def describe_image():
    """API لوصف الصورة"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'لم يتم إرسال صورة'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'لم يتم اختيار ملف'}), 400
        
        # قراءة الصورة
        image = Image.open(file.stream).convert('RGB')
        
        # وصف الصورة باللغتين
        english_desc = describe_image_english(image)
        arabic_desc = describe_image_arabic(image)
        
        return jsonify({
            'english': english_desc,
            'arabic': arabic_desc,
            'success': True
        })
    
    except Exception as e:
        return jsonify({'error': f'خطأ في معالجة الصورة: {str(e)}'}), 500

@app.route('/api/describe_url', methods=['POST'])
def describe_image_url():
    """API لوصف الصورة من رابط URL"""
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({'error': 'لم يتم إرسال رابط URL'}), 400
        
        url = data['url']
        
        # تحميل الصورة من الرابط
        response = requests.get(url)
        response.raise_for_status()
        
        # قراءة الصورة
        image = Image.open(BytesIO(response.content)).convert('RGB')
        
        # وصف الصورة باللغتين
        english_desc = describe_image_english(image)
        arabic_desc = describe_image_arabic(image)
        
        return jsonify({
            'english': english_desc,
            'arabic': arabic_desc,
            'success': True
        })
    
    except Exception as e:
        return jsonify({'error': f'خطأ في معالجة الصورة: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
