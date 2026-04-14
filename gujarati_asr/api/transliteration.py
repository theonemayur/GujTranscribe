"""
Gujlish → Gujarati Transliteration Pipeline
Production-level converter with 4-stage system
"""

import re
from typing import Dict, List, Optional

class GujlishConverter:
    def __init__(self):
        self._init_word_dict()
        self._init_expansions()
        self._init_phonetics()
        self._init_place_names()
        self._init_common_words()
    
    def _init_word_dict(self):
        """Complete word dictionary"""
        self.word_dict = {
            # Pronouns
            'hu': 'હું', 'h': 'હું',
            'chu': 'છું', 'chhu': 'છું',
            'che': 'છે', 'chhe': 'છે',
            'cho': 'છો', 'chho': 'છો',
            'tame': 'તમે', 'tam': 'તમે', 'tamaru': 'તમારુ', 'tamru': 'તમરુ', 'tamaro': 'તમારો', 'tamari': 'તમારી',
            'maru': 'મારુ', 'maaru': 'મારુ', 'maro': 'મારો', 'maare': 'મારે', 'mari': 'મારી',
            'apnu': 'પોતાનુ', 'apna': 'પોતાનુ', 'apni': 'પોતાની',
            'enu': 'એનુ', 'ena': 'એનો', 'eni': 'એની',
            'mane': 'મને', 'mne': 'મને',
            'saaru': 'સારું', 'sarhu': 'સારું', 'saru': 'સારું', 'saar': 'સારું', 'saarun': 'સારું',
            'avu': 'આવુ', 'avyo': 'આવ્યો', 'avyu': 'આવ્યું', 'avta': 'આવતો', 'avti': 'આવતી',
            'tena': 'તેના', 'teni': 'તેની', 'tenu': 'તેનુ',
            'mena': 'મેના', 'meni': 'મેની', 'menu': 'મેનુ',
            
            # Common words
            'naam': 'નામ', 'naa': 'નામ', 'nam': 'નામ', 'name': 'નામ',
            'mayur': 'મયુર', 'myur': 'મયુર',
            'kem': 'કેમ', 'kevu': 'કેવુ', 'kemar': 'કેમ',
            'shu': 'શુ', 'su': 'શું',
            'ek': 'એક', 'evu': 'એક', 'eku': 'એક',
            'bija': 'બીજો', 'biju': 'બીજો', 'bij': 'બીજો',
            'tri': 'ત્રણ', 'teen': 'ત્રણ', 'char': 'ચાર', 'chaar': 'ચાર', 'paanch': 'પાંચ',
            'hajar': 'હાજર', 'hato': 'હતો', 'hati': 'હતી', 'hatu': 'હતું',
            'darbar': 'દરબાર', 'darbaar': 'દરબાર',
            'raaj': 'રાજ', 'raj': 'રાજ', 'maharaj': 'મહારાજ',
            'desh': 'દેશ',
            'gujrat': 'ગુજરાત', 'gujarat': 'ગુજરાત', 'gujrati': 'ગુજરાતી',
            'india': 'ભારત', 'bharat': 'ભારત',
            'rahelu': 'રહેલું', 'rahel': 'રહેલ', 'rahe': 'રહે', 'rahyo': 'રહ્યું',
            'karu': 'કરુ', 'kryo': 'કર્યો', 'kryu': 'કર્યું', 'karta': 'કરતો', 'karti': 'કરતી', 'kare': 'કરે',
            'jao': 'જાઓ', 'javo': 'જવુ', 'jayo': 'જ્યો', 'jayu': 'જ્યું', 'jai': 'જાય',
            'thai': 'થઈ', 'thayu': 'થયું', 'thayo': 'થયો', 'thay': 'થય',
            'gai': 'ગઈ', 'gayu': 'ગયું', 'gayo': 'ગયો', 'gay': 'ગય',
            'bai': 'બઈ', 'bayu': 'બયું', 'bayo': 'બયો',
            'kya': 'ક્યાં', 'kyare': 'ક્યારે',
            
            # Verbs & particles
            'to': 'તો', 'mate': 'માટે', 'ne': 'ને', 'na': 'ના', 'no': 'નો', 'nu': 'નું',
            'ke': 'કે', 'kahe': 'કે', 'ka': 'કા', 'ki': 'કી', 'ku': 'કુ', 'ko': 'કો',
            'e': 'એ', 'aa': 'આ', 'u': 'ઉ', 'i': 'ઇ', 'o': 'ઓ',
            'je': 'જે', 'te': 'તે', 'eni': 'એની',
            'badhhu': 'બધું', 'sab': 'બધું', 'sabhu': 'બધું', 'badhu': 'બધું',
            'koi': 'કોઈ', 'keva': 'કેવા',
            'talu': 'તલુ', 'talav': 'તલાવ', 'samay': 'સમય',
            'bhai': 'ભાઈ', 'bahen': 'બહિન', 'bahin': 'બહિન',
            'student': 'વિદ્યાર્થી', 'teacher': 'શિક્ષક',
            'doctor': 'ડૉક્ટર',
            'rupees': 'રુપિયે', 'rupiya': 'રુપિયે', 'paise': 'પૈસે',
            'akbar': 'અકબર', 'birbal': 'બીરબલ',
            'ni': 'ની', 'na': 'ના', 'no': 'નો', 'nu': 'નું',
            'ma': 'મા', 'maa': 'માં',
            'developer': 'ડેવલપર',
            'engineer': 'એંજિનીર',
            'programmer': 'પ્રોગ્રામર',
        }
    
    def _init_expansions(self):
        """Incomplete word expansions - CRITICAL for accuracy"""
        self.expansions = {
            'mne': 'mane', 'mn': 'mane',
            'tme': 'tame', 'tm': 'tame',
            'kr': 'kar', 'kR': 'kar',
            'kro': 'karo',
            'pr': 'par',
            'mt': 'mat',
            'ch': 'chhu', 'chh': 'chhu', 'chu': 'chhu',
            'sh': 'shu',
            'h': 'hu',
            'v': 'vu', 'w': 'vu',
            'j': 'ju',
            's': 'su',
            'n': 'no',
            'd': 'du',
            'b': 'bu',
            'p': 'pu',
            't': 'tu',
            'k': 'ku',
        }
    
    def _init_phonetics(self):
        """Phoneme to Gujarati mapping"""
        self.consonants = {
            'k': 'ક', 'kh': 'ખ', 'g': 'ગ', 'gh': 'ઘ', 'ng': 'ઙ',
            'c': 'ચ', 'ch': 'ચ', 'chh': 'છ', 'jh': 'ઝ', 'j': 'જ', 'ny': 'ઞ',
            'tt': 'ટ', 'th': 'થ', 't': 'ત', 'd': 'દ', 'dh': 'ધ', 'n': 'ન',
            'p': 'પ', 'ph': 'ફ', 'f': 'ફ', 'b': 'બ', 'bh': 'ભ', 'm': 'મ',
            'y': 'ય', 'r': 'ર', 'l': 'લ', 'v': 'વ', 'w': 'વ',
            'sh': 'શ', 'shh': 'ષ', 's': 'સ', 'h': 'હ',
            'ks': 'ક્ષ', 'gn': 'જ્ઞ', 'q': 'ક', 'x': 'ક્સ', 'z': 'જ',
        }
        
        self.vowels = {
            'aa': 'ા', 'A': 'ા',
            'ee': 'ી', 'ii': 'ી', 'I': 'ી',
            'oo': 'ૂ', 'uu': 'ૂ', 'U': 'ૂ',
            'ai': 'ૈ', 'ei': 'એ', 'e': 'ે', 'E': 'ે',
            'au': 'ૌ', 'ou': 'ઓ', 'o': 'ો', 'O': 'ો',
            'a': 'અ', 'i': 'િ', 'u': 'ુ',
            'ri': 'ઋ', 'ri': 'ૃ',
        }
    
    def _init_place_names(self):
        """Gujarati city and place names"""
        self.place_names = {
            'ahmedabad': 'અમદાવાદ', 'ahm': 'અમદાવાદ', 'amd': 'અમદાવાદ',
            'vadodara': 'વડોદરા', 'baroda': 'વડોદરા',
            'surat': 'સુરત',
            'rajkot': 'રાજકોટ',
            'gandhinagar': 'ગાંધીનગર',
            'jamnagar': 'જામનગર',
            'junagadh': 'જૂનાગઢ',
            'bhavnagar': 'ભાવનગર',
            'anand': 'આનંદ',
            'nadiad': 'નડિયાદ',
            'morbi': 'મોરબી',
            'mehsana': 'મહેસાણા',
            'patan': 'પાટણ',
            'valsad': 'વલ્સાદ',
            'vapi': 'વાપી',
            'daman': 'દમણ',
            'dwarka': 'દ્વારકા',
            'porbandar': 'પોરબંદર',
            'kutch': 'કચ્છ', 'kch': 'કચ્છ',
        }
    
    def _init_common_words(self):
        """Additional common words"""
        self.common_words = {
            'minute': 'મિનિટ', 'min': 'મિનિટ',
            'divas': 'દિવસ', 'div': 'દિવસ', 'din': 'દિવસ',
            'vanchu': 'વાંછુ', 'vanchhi': 'વાંછ્યે', 'vanchha': 'વાંછ્યો',
            'joiye': 'જોઈએ', 'joi': 'જોઈ',
            'yaha': 'અહીં', 'yah': 'અહીં', 'yi': 'આ',
            'kaam': 'કામ', 'kam': 'કામ',
            'mat': 'માટે', 'mate': 'માટે',
            'kya': 'ક્યાં', 'kyare': 'ક્યારે',
        }
        self.word_dict.update(self.common_words)
        self.word_dict.update(self.place_names)
    
    def _is_english_word(self, word: str) -> bool:
        """Check if word is purely English (preserve only very common English words)"""
        if not word or len(word) < 3:
            return False
        
        w = word.lower()
        
        # Only preserve very specific, common English words
        # Do NOT preserve technical terms like developer, student - those should convert
        english_words = {
            # Common greetings
            'hello', 'hi', 'hey', 'thanks', 'thank', 'please', 'sorry',
            # Common phrases
            'yes', 'no', 'ok', 'good', 'bad', 'nice', 'great',
            # Common verbs
            'see', 'come', 'go', 'get', 'make', 'take',
            # Determiners/prepositions
            'the', 'is', 'are', 'was', 'were', 'this', 'that', 'these', 'those',
        }
        
        if w in english_words:
            return True
        
        return False
    
    def normalize(self, text: str) -> str:
        """Stage 1: Normalize Gujlish"""
        words = text.split()
        result = []
        
        for word in words:
            # Skip if already looks like English (preserve as-is)
            if self._is_english_word(word):
                result.append(word)
                continue
            
            # Preserve known acronyms
            if word.upper() in ['AI', 'ML', 'NLP', 'API', 'URL', 'HTTP', 'OK', 'USB', 'LED', 'TV']:
                result.append(word.upper())
                continue
            
            word_clean = re.sub(r'[^\w]', '', word)
            if not word_clean:
                result.append(word)
                continue
            
            # Apply expansions
            if word_clean in self.expansions:
                word_clean = self.expansions[word_clean]
            
            # Regex expansions
            word_clean = re.sub(r'^m(n|N)(e|E)?$', 'mane', word_clean)
            word_clean = re.sub(r'^t(m|M)(e|E)?$', 'tame', word_clean)
            word_clean = re.sub(r'^ch(h|hu)?$', 'chhu', word_clean)
            word_clean = re.sub(r'che$', 'chhe', word_clean)
            word_clean = re.sub(r'chu$', 'chhu', word_clean)
            word_clean = re.sub(r'cho$', 'chho', word_clean)
            word_clean = re.sub(r'shu$', 'shu', word_clean)
            word_clean = re.sub(r'hu$', 'hu', word_clean)
            
            result.append(word_clean)
        
        return ' '.join(result)
    
    def transliterate(self, text: str) -> str:
        """Stage 2: Transliterate"""
        words = text.split()
        result = []
        
        for word in words:
            # Skip words that were identified as English in normalize
            if self._is_english_word(word):
                result.append(word)
                continue
            
            guj = self._convert_word(word)
            result.append(guj)
        
        return ' '.join(result)
    
    def _convert_word(self, word: str) -> str:
        """Convert single word"""
        w = word.lower()
        
        # Direct dictionary match
        if w in self.word_dict:
            return self.word_dict[w]
        
        # Place names
        if w in self.place_names:
            return self.place_names[w]
        
        # Handle nu/na/ni/no endings (possessive)
        for suffix, guj in [('nu', 'નું'), ('na', 'ના'), ('ni', 'ની'), ('no', 'નો')]:
            if w.endswith(suffix) and len(w) > len(suffix) + 2:
                base = w[:-len(suffix)]
                if base in self.word_dict:
                    return self.word_dict[base] + guj
                base_guj = self._phonetic_convert(base)
                return base_guj + guj if base_guj else word
        
        # Handle verb forms: che/chu/cho
        if w.endswith('che') or w.endswith('chhe'):
            base = w[:-3] if w.endswith('che') else w[:-4]
            if base in self.word_dict:
                return self.word_dict[base] + 'છે'
            base_guj = self._phonetic_convert(base)
            return base_guj + 'છે' if base_guj else word
        
        if w.endswith('cho') or w.endswith('chho'):
            base = w[:-3] if w.endswith('cho') else w[:-4]
            if base in self.word_dict:
                return self.word_dict[base] + 'છો'
            base_guj = self._phonetic_convert(base)
            return base_guj + 'છો' if base_guj else word
        
        if w.endswith('chu') or w.endswith('chhu'):
            base = w[:-3] if w.endswith('chu') else w[:-4]
            if base in self.word_dict:
                return self.word_dict[base] + 'છું'
            base_guj = self._phonetic_convert(base)
            return base_guj + 'છું' if base_guj else word
        
        return self._phonetic_convert(w)
    
    def _phonetic_convert(self, word: str) -> str:
        """Phonetic conversion"""
        if not word:
            return ''
        
        result = ''
        i = 0
        w = word.lower()
        
        clusters = ['chh', 'shh', 'gh', 'dh', 'kh', 'ph', 'bh', 'th', 'sh', 'ks', 'gn', 'ng', 'ny']
        
        while i < len(w):
            matched = False
            
            for cluster in clusters:
                if w[i:].startswith(cluster):
                    result += self.consonants.get(cluster, '')
                    i += len(cluster)
                    matched = True
                    
                    if i < len(w):
                        v = self._get_vowel(w[i:])
                        if v:
                            result += v['char']
                            i += v['len']
                    break
            
            if matched:
                continue
            
            c = w[i]
            
            if c in self.consonants:
                result += self.consonants[c]
                i += 1
                
                if i < len(w):
                    v = self._get_vowel(w[i:])
                    if v:
                        result += v['char']
                        i += v['len']
                    elif w[i] in 'aeiou':
                        pass
                    else:
                        result += 'અ'
            elif c in 'aeiou':
                v = self._get_vowel(w[i:])
                if v:
                    result += v['char']
                    i += v['len']
            else:
                i += 1
        
        return result
    
    def _get_vowel(self, text: str) -> Optional[Dict]:
        """Extract vowel from text"""
        for pattern, char in [
            ('aa', 'ા'), ('ee', 'ી'), ('oo', 'ૂ'), ('ii', 'ી'), ('uu', 'ૂ'),
            ('ai', 'ૈ'), ('au', 'ૌ'), ('ei', 'એ'),
            ('a', 'અ'), ('i', 'િ'), ('u', 'ુ'), ('e', 'ે'), ('o', 'ો'),
        ]:
            if text.startswith(pattern):
                return {'char': char, 'len': len(pattern)}
        return None
    
    def post_correct(self, text: str) -> str:
        """Stage 3: Post-correct"""
        text = re.sub(r'ંં', 'ં', text)
        
        fixes = [
            ('ચે', 'છે'),
            ('સારુંં', 'સારું'),
            ('સારં', 'સારું'),
            ('મનઇ', 'મને'),
            ('મનૅ', 'મને'),
            ('ુંં', 'ં'),
        ]
        
        for wrong, correct in fixes:
            text = text.replace(wrong, correct)
        
        return text
    
    def convert(self, text: str) -> str:
        """Main pipeline"""
        if not text:
            return ''
        
        norm = self.normalize(text)
        trans = self.transliterate(norm)
        final = self.post_correct(trans)
        
        return final
    
    def suggest(self, word: str) -> List[str]:
        """Suggest corrections"""
        sug = []
        w = word.lower()
        
        if w in self.expansions:
            sug.append(self.expansions[w])
        
        if w.startswith('mn'):
            sug.append('mane')
        if w.startswith('tm'):
            sug.append('tame')
        if 'ch' in w and 'hu' not in w:
            sug.append(w.replace('ch', 'chhu'))
        
        return list(set(sug))[:3]


def convert_text(text: str) -> str:
    """Convenience function"""
    converter = GujlishConverter()
    return converter.convert(text)


def gujarati_to_gujlish(text: str) -> str:
    """Convert Gujarati to Gujlish (reverse)"""
    guj_to_eng = {
        'અ': 'a', 'આ': 'aa', 'ઇ': 'i', 'ઈ': 'ee', 'ઉ': 'u', 'ઊ': 'oo', 'ઋ': 'ri',
        'એ': 'e', 'ઐ': 'ai', 'ઓ': 'o', 'ઔ': 'au',
        'ક': 'k', 'ખ': 'kh', 'ગ': 'g', 'ઘ': 'gh', 'ઙ': 'ng',
        'ચ': 'ch', 'છ': 'chh', 'જ': 'j', 'ઝ': 'jh', 'ઞ': 'ny',
        'ત': 't', 'થ': 'th', 'દ': 'd', 'ધ': 'dh', 'ન': 'n',
        'પ': 'p', 'ફ': 'ph', 'બ': 'b', 'ભ': 'bh', 'મ': 'm',
        'ય': 'y', 'ર': 'r', 'લ': 'l', 'વ': 'v', 'શ': 'sh', 'ષ': 'sh', 'સ': 's', 'હ': 'h',
        '્': '', 'ા': 'aa', 'િ': 'i', 'ી': 'ee', 'ુ': 'u', 'ૂ': 'oo',
        'ે': 'e', 'ૈ': 'ai', 'ો': 'o', 'ૌ': 'au', 'ં': 'n',
    }
    
    result = ''
    keys = sorted(guj_to_eng.keys(), key=lambda x: -len(x))
    
    i = 0
    while i < len(text):
        matched = False
        for key in keys:
            if text[i:].startswith(key):
                result += guj_to_eng[key]
                i += len(key)
                matched = True
                break
        if not matched:
            result += text[i]
            i += 1
    
    return result


if __name__ == '__main__':
    c = GujlishConverter()
    
    tests = [
        'mne saaru che',
        'hu avu chu',
        'kem cho',
        'tame kya jao cho',
    ]
    
    for t in tests:
        print(f'{t} -> {c.convert(t)}')