var wordMap = {
    'tamaru': 'તમારુ', 'tamru': 'તમરુ', 'tamaro': 'તમારો', 'tamari': 'તમારી', 'tamara': 'તમારા',
    'maru': 'મારુ', 'maaru': 'મારુ', 'maro': 'મારો', 'maare': 'મારે', 'mari': 'મારી',
    'naam': 'નામ', 'naa': 'નામ', 'nam': 'નામ',
    'mayur': 'મયુર', 'myur': 'મયુર',
    'chhu': 'છું', 'chu': 'છુ',
    'chhe': 'છે', 'che': 'છે',
    'hu': 'હું',
    'tame': 'તમે', 'tam': 'તમે',
    'kem': 'કેમ', 'kevu': 'કેમ',
    'cho': 'છો',
    'akbar': 'અકબર',
    'birbal': 'બીરબલ',
    'ni': 'ની', 'na': 'ના',
    'vat': 'વાત', 'vaat': 'વાત',
    'ek': 'એક',
    'bija': 'બીજો',
    'tri': 'ત્રણ',
    'char': 'ચાર',
    'paanch': 'પાંચ',
    'hajar': 'હાજર', 'hato': 'હતો', 'hati': 'હતી',
    'darbar': 'દરબાર',
    'raaj': 'રાજ',
    'maharaj': 'મહારાજ',
    'desh': 'દેશ',
    'gujrat': 'ગુજરાત', 'gujarat': 'ગુજરાત',
    'gujratma': 'ગુજરાતમાં', 'ma': 'માં',
    'rahelu': 'રહેલું', 'rahe': 'રહે',
    'programmer': 'પ્રોગ્રામર', 'developer': 'ડેવલપર',
    'student': 'વિદ્યાર્થી', 'teacher': 'શિક્ષક',
    'rupees': 'રુપિયે', 'paise': 'પૈસે',
    'badhhu': 'બધું', 'sab': 'બધું',
    'to': 'તો', 'mate': 'માટે',
    'ne': 'ને',
    'ke': 'કે',
    'guj': 'ગુજ',
    'rat': 'રત',
    'ra': 'ર',
    'lo': 'લો', 'la': 'લ',
    'wo': 'વો', 'wa': 'વા',
    'yo': 'યો', 'ya': 'યા',
    'ho': 'હો', 'ha': 'હા',
    'go': 'ગો', 'ga': 'ગા',
    'ko': 'કો', 'ka': 'કા',
    'jo': 'જો', 'ja': 'જા',
    'to': 'તો', 'ta': 'તા',
    'do': 'દો', 'da': 'દા',
    'po': 'પો', 'pa': 'પા',
    'bo': 'બો', 'ba': 'બા',
    'mo': 'મો', 'ma': 'મા',
    'so': 'સો', 'sa': 'સા',
    'khal': 'ખાલિ',
    'sau': 'સો'
};

function gujlishToGujarati(input) {
    var text = ' ' + input.toLowerCase().trim() + ' ';
    
    var words = text.split(/\s+/);
    var result = [];
    
    for (var w = 0; w < words.length; w++) {
        var word = words[w];
        
        if (wordMap[word]) {
            result.push(wordMap[word]);
            continue;
        }
        
        var bestMatch = '';
        var bestLen = 0;
        
        for (var key in wordMap) {
            if (word === key || word === key + 'e' || word === key + 'u' || word === key + 'o') {
                if (key.length > bestLen) {
                    bestMatch = wordMap[key];
                    bestLen = key.length;
                }
            }
        }
        
        if (bestMatch) {
            result.push(bestMatch);
        } else {
            result.push(word);
        }
    }
    
    return result.join(' ');
}

console.log('tamaru naam chu che => ' + gujlishToGujarati('tamaru naam chu che'));
console.log('maru naam mayur che => ' + gujlishToGujarati('maru naam mayur che'));
console.log('kem cho => ' + gujlishToGujarati('kem cho'));
console.log('hu gujrat ma rahelu chhu => ' + gujlishToGujarati('hu gujrat ma rahelu chhu'));
