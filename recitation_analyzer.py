import re

class RecitationAnalyzer:
    def __init__(self):
        self.tashkeel_pattern = re.compile(
            r'[\u0617-\u061A\u064B-\u0652\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED]'
        )

    def strip_tashkeel(self, text):
        return self.tashkeel_pattern.sub('', text)
    
    # ✅ تصحيح 5: تحسين دالة normalize
    def normalize(self, text):
        """Normalize Arabic text with proper Hamza/Yaa unification."""
        # إزالة التطويل
        text = re.sub(r'ـ', '', text)
        # توحيد المسافات
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def normalize_for_comparison(self, text):
        """تطبيع أعمق للمقارنة (بدون تشكيل + توحيد الهمزات)."""
        text = self.normalize(text)
        text = self.strip_tashkeel(text)
        # ✅ توحيد الهمزات
        text = re.sub(r'[أإآ]', 'ا', text)
        # ✅ توحيد الياء
        text = text.replace('ى', 'ي')
        # ✅ التاء المربوطة
        text = text.replace('ة', 'ه')
        return text

    def analyze(self, transcription, expected_text):
        """تحليل التلاوة باستخدام Needleman-Wunsch."""
        if not expected_text:
            expected_words = []
        else:
            expected_words = self.normalize(expected_text).split()
            
        if not transcription:
            recited_words = []
        else:
            recited_words = self.normalize(transcription).split()
            
        n = len(expected_words)
        m = len(recited_words)
        
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        
        for i in range(n + 1):
            dp[i][0] = -i
        for j in range(m + 1):
            dp[0][j] = -j
            
        def get_score(exp_w, rec_w):
            # ✅ مقارنة كاملة (تطابق تام)
            if exp_w == rec_w:
                return 2
            # ✅ مقارنة بدون تشكيل
            elif self.strip_tashkeel(exp_w) == self.strip_tashkeel(rec_w):
                return 1
            # ✅ مقارنة عميقة (مع توحيد الهمزات والياء)
            elif self.normalize_for_comparison(exp_w) == self.normalize_for_comparison(rec_w):
                return 1  # نعتبره خطأ تشكيل
            return -1

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                match = dp[i-1][j-1] + get_score(expected_words[i-1], recited_words[j-1])
                delete = dp[i-1][j] - 1
                insert = dp[i][j-1] - 1
                dp[i][j] = max(match, delete, insert)
                
        # Backtracking
        i, j = n, m
        alignment = []
        
        while i > 0 or j > 0:
            if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + get_score(expected_words[i-1], recited_words[j-1]):
                exp_w = expected_words[i-1]
                rec_w = recited_words[j-1]
                
                if exp_w == rec_w:
                    status = "correct"
                # ✅ تشكيل غلط
                elif self.strip_tashkeel(exp_w) == self.strip_tashkeel(rec_w):
                    status = "tashkeel_error"
                # ✅ خطأ في الهمزة أو حرف بسيط
                elif self.normalize_for_comparison(exp_w) == self.normalize_for_comparison(rec_w):
                    status = "tashkeel_error"
                else:
                    status = "substitution"
                
                alignment.append({
                    "expected_index": i - 1,
                    "expected_word": exp_w,
                    "recited_word": rec_w,
                    "status": status
                })
                i -= 1
                j -= 1
            elif i > 0 and (j == 0 or dp[i][j] == dp[i-1][j] - 1):
                alignment.append({
                    "expected_index": i - 1,
                    "expected_word": expected_words[i-1],
                    "recited_word": None,
                    "status": "missing"
                })
                i -= 1
            else:
                alignment.append({
                    "expected_index": None,
                    "expected_word": None,
                    "recited_word": recited_words[j-1],
                    "status": "extra"
                })
                j -= 1
                
        alignment.reverse()
        
        # Summary
        summary = {
            "total_expected": n,
            "correct": 0,
            "tashkeel_errors": 0,
            "substitutions": 0,
            "missing": 0,
            "extra": 0,
            "accuracy": 0.0
        }
        
        for w in alignment:
            if w["status"] == "correct":
                summary["correct"] += 1
            elif w["status"] == "tashkeel_error":
                summary["tashkeel_errors"] += 1
            elif w["status"] == "substitution":
                summary["substitutions"] += 1
            elif w["status"] == "missing":
                summary["missing"] += 1
            elif w["status"] == "extra":
                summary["extra"] += 1
                
        if n > 0:
            acc = (summary["correct"] + (summary["tashkeel_errors"] * 0.5)) / n
            summary["accuracy"] = float(min(1.0, max(0.0, acc)))
            
        return {
            "words": alignment,
            "summary": summary
        }
