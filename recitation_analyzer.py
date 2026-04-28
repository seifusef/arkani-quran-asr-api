import re

class RecitationAnalyzer:
    def __init__(self):
        # Match all Arabic diacritics (Tashkeel)
        self.tashkeel_pattern = re.compile(
            r'[\u0617-\u061A\u064B-\u0652\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED]'
        )

    def strip_tashkeel(self, text):
        """Remove all Arabic diacritics from text."""
        return self.tashkeel_pattern.sub('', text)
        
    def normalize(self, text):
        """Normalize Arabic text."""
        # إزالة التطويل
        text = re.sub(r'ـ', '', text)
        # توحيد المسافات
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def get_tashkeel_diff(self, expected_word, recited_word):
        """
        تحديد فرق التشكيل بالتفصيل (مفيد للمستخدم).
        مثال: أَنْعَمْتَ vs أَنْعَمْتُ → "الحرف الأخير: فتحة بدل ضمة"
        """
        if len(expected_word) != len(recited_word):
            return "اختلاف في الحركات"
        
        diffs = []
        for i, (e_char, r_char) in enumerate(zip(expected_word, recited_word)):
            if e_char != r_char:
                # الحرف الأساسي قبل الاختلاف
                base_char_idx = i - 1 if i > 0 else 0
                base_char = self.strip_tashkeel(expected_word[base_char_idx]) if base_char_idx < len(expected_word) else ""
                diffs.append(f"حركة على '{base_char}'")
        
        return " - ".join(diffs) if diffs else "اختلاف بسيط في التشكيل"

    def analyze(self, transcription, expected_text):
        """
        Word-level alignment using Needleman-Wunsch.
        """
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
            if exp_w == rec_w:
                return 2
            elif self.strip_tashkeel(exp_w) == self.strip_tashkeel(rec_w):
                return 1
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
                    detail = None
                elif self.strip_tashkeel(exp_w) == self.strip_tashkeel(rec_w):
                    status = "tashkeel_error"
                    detail = self.get_tashkeel_diff(exp_w, rec_w)
                else:
                    status = "substitution"
                    detail = None
                
                word_data = {
                    "expected_index": i - 1,
                    "expected_word": exp_w,
                    "recited_word": rec_w,
                    "status": status
                }
                if detail:
                    word_data["detail"] = detail
                
                alignment.append(word_data)
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
            "accuracy": 0.0,
            "tashkeel_accuracy": 0.0,  # دقة التشكيل بشكل منفصل
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
            # دقة عامة (تشكيل = نص نقطة)
            acc = (summary["correct"] + (summary["tashkeel_errors"] * 0.5)) / n
            summary["accuracy"] = float(min(1.0, max(0.0, acc)))
            
            # دقة التشكيل بس (للكلمات اللي قيلت بشكل صحيح)
            words_with_tashkeel = summary["correct"] + summary["tashkeel_errors"]
            if words_with_tashkeel > 0:
                summary["tashkeel_accuracy"] = float(summary["correct"] / words_with_tashkeel)
            
        return {
            "words": alignment,
            "summary": summary
        }
