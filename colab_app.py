# تطبيق وصف الصور الذكي - سوريا 🇸🇾
# يمكن تشغيل هذا الملف مباشرة في Google Colab

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
import requests
from io import BytesIO
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

# قالب HTML مع هوية بصرية مستوحاة من الهوية البصرية الجديدة لسوريا
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>وصف الصور الذكي - سوريا</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;500;600;700;800;900&display=swap');
        
        body {
            font-family: 'Cairo', sans-serif;
            background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 25%, #3b82f6 50%, #1e40af 75%, #1e3a8a 100%);
            min-height: 100vh;
        }
        
        .syria-gradient {
            background: linear-gradient(135deg, #dc2626 0%, #ea580c 25%, #f59e0b 50%, #16a34a 75%, #0891b2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .syria-border {
            border: 3px solid transparent;
            background: linear-gradient(45deg, #dc2626, #ea580c, #f59e0b, #16a34a, #0891b2) border-box;
            border-radius: 15px;
            background-clip: padding-box, border-box;
        }
        
        .syria-shadow {
            box-shadow: 0 10px 30px rgba(220, 38, 38, 0.3), 
                        0 5px 15px rgba(234, 88, 12, 0.3),
                        0 0 20px rgba(245, 158, 11, 0.2);
        }
        
        .upload-area {
            background: linear-gradient(135deg, rgba(30, 58, 138, 0.1) 0%, rgba(59, 130, 246, 0.1) 100%);
            border: 2px dashed #3b82f6;
            transition: all 0.3s ease;
        }
        
        .upload-area:hover {
            border-color: #dc2626;
            background: linear-gradient(135deg, rgba(220, 38, 38, 0.1) 0%, rgba(59, 130, 246, 0.1) 100%);
        }
        
        .upload-area.dragover {
            border-color: #16a34a;
            background: linear-gradient(135deg, rgba(22, 163, 74, 0.2) 0%, rgba(59, 130, 246, 0.2) 100%);
            transform: scale(1.02);
        }
        
        .result-card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .loading-spinner {
            border: 4px solid rgba(255, 255, 255, 0.3);
            border-top: 4px solid #dc2626;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .fade-in {
            animation: fadeIn 0.5s ease-in;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body class="antialiased">
    <!-- Header -->
    <header class="bg-white/10 backdrop-blur-md border-b border-white/20">
        <div class="container mx-auto px-4 py-6">
            <div class="flex flex-col md:flex-row items-center justify-between">
                <div class="text-center md:text-right mb-4 md:mb-0">
                    <h1 class="text-4xl md:text-5xl font-bold syria-gradient">
                        وصف الصور الذكي
                    </h1>
                    <p class="text-white/80 text-lg mt-2">تطبيق ذكي لوصف الصور باللغتين العربية والإنجليزية</p>
                </div>
                <div class="flex items-center space-x-4 space-x-reverse">
                    <div class="w-16 h-16 bg-gradient-to-br from-red-600 to-blue-600 rounded-full flex items-center justify-center">
                        <span class="text-white text-2xl font-bold">س</span>
                    </div>
                </div>
            </div>
        </div>
    </header>

    <!-- Main Content -->
    <main class="container mx-auto px-4 py-8">
        <!-- Upload Section -->
        <section class="max-w-4xl mx-auto mb-12">
            <div class="text-center mb-8">
                <h2 class="text-3xl font-bold text-white mb-4">ارفع صورة لوصفها</h2>
                <p class="text-white/80 text-lg">يمكنك رفع صورة من جهازك أو إدخال رابط URL</p>
            </div>
            
            <!-- File Upload -->
            <div class="upload-area rounded-2xl p-8 text-center mb-6" id="uploadArea">
                <div class="mb-4">
                    <svg class="w-16 h-16 mx-auto text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
                    </svg>
                </div>
                <p class="text-white text-lg mb-4">اسحب وأفلت الصورة هنا أو انقر للاختيار</p>
                <input type="file" id="imageInput" accept="image/*" class="hidden">
                <button onclick="document.getElementById('imageInput').click()" class="bg-gradient-to-r from-red-600 to-blue-600 text-white px-6 py-3 rounded-full font-semibold hover:from-red-700 hover:to-blue-700 transition-all duration-300 transform hover:scale-105">
                    اختر صورة
                </button>
            </div>
            
            <!-- URL Input -->
            <div class="bg-white/10 backdrop-blur-md rounded-2xl p-6 mb-6">
                <h3 class="text-xl font-semibold text-white mb-4 text-center">أو أدخل رابط URL للصورة</h3>
                <div class="flex flex-col sm:flex-row gap-4">
                    <input type="url" id="imageUrl" placeholder="https://example.com/image.jpg" class="flex-1 px-4 py-3 rounded-lg bg-white/20 border border-white/30 text-white placeholder-white/60 focus:outline-none focus:border-blue-400 focus:ring-2 focus:ring-blue-400/50">
                    <button onclick="describeImageFromUrl()" class="bg-gradient-to-r from-green-600 to-blue-600 text-white px-6 py-3 rounded-lg font-semibold hover:from-green-700 hover:to-blue-700 transition-all duration-300">
                        وصف الصورة
                    </button>
                </div>
            </div>
        </section>

        <!-- Results Section -->
        <section class="max-w-4xl mx-auto" id="resultsSection" style="display: none;">
            <div class="text-center mb-8">
                <h2 class="text-3xl font-bold text-white mb-4">نتائج الوصف</h2>
            </div>
            
            <div class="grid md:grid-cols-2 gap-6">
                <!-- English Description -->
                <div class="result-card rounded-2xl p-6 syria-shadow">
                    <div class="flex items-center mb-4">
                        <div class="w-8 h-8 bg-gradient-to-br from-blue-600 to-purple-600 rounded-full flex items-center justify-center mr-3">
                            <span class="text-white text-sm font-bold">EN</span>
                        </div>
                        <h3 class="text-xl font-semibold text-white">الوصف بالإنجليزية</h3>
                    </div>
                    <p id="englishResult" class="text-white/90 text-lg leading-relaxed"></p>
                </div>
                
                <!-- Arabic Description -->
                <div class="result-card rounded-2xl p-6 syria-shadow">
                    <div class="flex items-center mb-4">
                        <div class="w-8 h-8 bg-gradient-to-br from-red-600 to-orange-600 rounded-full flex items-center justify-center mr-3">
                            <span class="text-white text-sm font-bold">ع</span>
                        </div>
                        <h3 class="text-xl font-semibold text-white">الوصف بالعربية</h3>
                    </div>
                    <p id="arabicResult" class="text-white/90 text-lg leading-relaxed"></p>
                </div>
            </div>
            
            <!-- Image Preview -->
            <div class="mt-8 text-center">
                <h3 class="text-xl font-semibold text-white mb-4">معاينة الصورة</h3>
                <div class="inline-block">
                    <img id="imagePreview" class="max-w-full h-auto rounded-2xl syria-shadow" alt="معاينة الصورة">
                </div>
            </div>
        </section>

        <!-- Loading Section -->
        <section id="loadingSection" class="max-w-4xl mx-auto text-center" style="display: none;">
            <div class="bg-white/10 backdrop-blur-md rounded-2xl p-12">
                <div class="loading-spinner mx-auto mb-6"></div>
                <h3 class="text-2xl font-semibold text-white mb-4">جاري معالجة الصورة...</h3>
                <p class="text-white/80 text-lg">يرجى الانتظار بينما يقوم الذكاء الاصطناعي بوصف الصورة</p>
            </div>
        </section>
    </main>

    <!-- Footer -->
    <footer class="bg-white/10 backdrop-blur-md border-t border-white/20 mt-16">
        <div class="container mx-auto px-4 py-8">
            <div class="text-center">
                <p class="text-white/60">&copy; 2024 تطبيق وصف الصور الذكي - سوريا</p>
                <p class="text-white/60 mt-2">مطور بأحدث تقنيات الذكاء الاصطناعي</p>
            </div>
        </div>
    </footer>

    <script>
        // File Upload Handling
        const uploadArea = document.getElementById('uploadArea');
        const imageInput = document.getElementById('imageInput');
        const resultsSection = document.getElementById('resultsSection');
        const loadingSection = document.getElementById('loadingSection');
        const imagePreview = document.getElementById('imagePreview');
        const englishResult = document.getElementById('englishResult');
        const arabicResult = document.getElementById('arabicResult');

        // Drag and Drop Events
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0 && files[0].type.startsWith('image/')) {
                handleImageUpload(files[0]);
            }
        });

        imageInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleImageUpload(e.target.files[0]);
            }
        });

        function handleImageUpload(file) {
            const formData = new FormData();
            formData.append('image', file);
            
            showLoading();
            describeImage(formData);
            
            // Show image preview
            const reader = new FileReader();
            reader.onload = (e) => {
                imagePreview.src = e.target.result;
            };
            reader.readAsDataURL(file);
        }

        async function describeImage(formData) {
            try {
                const response = await fetch('/api/describe', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.success) {
                    showResults(data.english, data.arabic);
                } else {
                    alert('خطأ: ' + data.error);
                    hideLoading();
                }
            } catch (error) {
                console.error('Error:', error);
                alert('حدث خطأ أثناء معالجة الصورة');
                hideLoading();
            }
        }

        async function describeImageFromUrl() {
            const url = document.getElementById('imageUrl').value.trim();
            if (!url) {
                alert('يرجى إدخال رابط URL صحيح');
                return;
            }

            showLoading();
            
            try {
                const response = await fetch('/api/describe_url', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ url: url })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    showResults(data.english, data.arabic);
                    // Show image preview from URL
                    imagePreview.src = url;
                } else {
                    alert('خطأ: ' + data.error);
                    hideLoading();
                }
            } catch (error) {
                console.error('Error:', error);
                alert('حدث خطأ أثناء معالجة الصورة');
                hideLoading();
            }
        }

        function showLoading() {
            loadingSection.style.display = 'block';
            resultsSection.style.display = 'none';
        }

        function hideLoading() {
            loadingSection.style.display = 'none';
        }

        function showResults(english, arabic) {
            hideLoading();
            englishResult.textContent = english;
            arabicResult.textContent = arabic;
            resultsSection.style.display = 'block';
            resultsSection.classList.add('fade-in');
        }
    </script>
</body>
</html>
'''

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
    print("🚀 بدء تشغيل تطبيق وصف الصور الذكي - سوريا")
    print("📱 التطبيق متاح على: http://localhost:5000")
    print("🌐 للتشغيل على Colab، استخدم ngrok:")
    print("   !pip install pyngrok")
    print("   from pyngrok import ngrok")
    print("   public_url = ngrok.connect(5000)")
    print("   print(f'التطبيق متاح على: {public_url}')")
    app.run(debug=True, host='0.0.0.0', port=5000)