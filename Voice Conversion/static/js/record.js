var imgpath1 = "/static/images/material/blank.jpg";
var imgpath2 = "/static/images/material/blank.jpg";
var wavepath1 = "/static/audio/blank.wav";
var wavepath2 = "/static/audio/blank.wav";
var recorder;
var audio = document.getElementById('#wave1');
var loadinggif = "/static/images/material/newloading.gif";

function page_jump(page_url) {
    window.location.href = page_url;
}

function convert() {
    var index = wavepath1.lastIndexOf("/");
    var length = wavepath1.length;
    var waveurl = wavepath1.substring(index + 1, length);
    // alert(waveurl);
    var fileLen = waveurl.lastIndexOf(".");
    var fileName = waveurl.substring(0, fileLen);
    var picpath = fileName + '.jpg';
    document.getElementById('result').src = loadinggif;
    $.get('/upload/convert', { 'method': 'GET', 'data': waveurl }, function success() {
        imgpath2 = "/static/images/convert/" + picpath;
        wavepath2 = "/static/audio/convert/" + waveurl;
        document.getElementById('result').src = imgpath2;
        var audio = document.getElementById('wave2');
        audio.src = wavepath2;
        audio.load();
    });// send the file name to the upload
}

function Play(id) {
    var audio = document.getElementById(id);
    audio.load();
    audio.play();
}

function preview(picname) {
    document.getElementById(picname).style.display = "block";
}

function out(picname) {
    document.getElementById(picname).style.display = "none";
}

function startRecording() {
    HZRecorder.get(
        function (rec) {
            recorder = rec;
            recorder.start();
        }, {
            sampleBits: 16,
            sampleRate: 16000
        }
    );
}
function stopRecording() {
    recorder.stop();
    var blob = recorder.getBlob();
    var url = URL.createObjectURL(blob);
    var hf = document.createElement('a');
    hf.href = url;
    var datestr = new Date().toISOString() + '.wav';
    var reg = new RegExp(':', "g")
    hf.download = datestr.replace(reg, '_');
    hf.innerHTML = hf.download;
    hf.click();
    var waveurl = hf.download;
    // alert(waveurl);
    var fileLen = waveurl.lastIndexOf(".");
    var fileName = waveurl.substring(0, fileLen);
    var picpath = fileName + '.jpg';
    // alert(picpath);
    document.getElementById('consignerRdSign').src = loadinggif;
    $.get('/recordsth', { 'method': 'GET', 'data': waveurl }, function success() {
        imgpath1 = "/static/images/" + picpath;
        wavepath1 = "/static/audio/" + waveurl;
        document.getElementById('consignerRdSign').src = imgpath1;
        var audio = document.getElementById('wave1');
        audio.src = wavepath1;
        audio.load();
        // alert(imgpath1);
    });// send the file name to the upload
}
function Play1() {
    Play('wave1');
}

function Play2() {
    Play("wave2");
}

function clearRecording() {
    imgpath1 = "/static/images/material/blank.jpg";
    imgpath2 = "/static/images/material/blank.jpg";
    wavepath1 = "/static/audio/blank.wav";
    wavepath2 = "/static/audio/blank.wav";
    document.getElementById('consignerRdSign').src = imgpath1;
    document.getElementById('result').src = imgpath2;
    var audio = document.getElementById('wave1');
    audio.src = wavepath1;
    audio.load();
    var audio2 = document.getElementById('wave2');
    audio2.src = wavepath2;
    audio2.load();
}