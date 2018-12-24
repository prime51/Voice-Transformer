var imgpath1 = "/static/images/material/blank.jpg";
var imgpath2 = "/static/images/material/blank.jpg";
var wavepath1 = "/static/audio/blank.wav";
var wavepath2 = "/static/audio/blank.wav";
var loadinggif = "/static/images/material/newloading.gif";

function page_jump(page_url) {
    window.location.href = page_url;
}

function F_Open_dialog(id) {
    document.getElementById(id).click();
}

function img1Change(e, imageid) {
    var reader = new FileReader();
    reader.onload = (function (file) {
        return function (e) { };
    })(e.target.files[0]);
    reader.readAsDataURL(e.target.files[0]);
    var waveurl = e.currentTarget.files[0].name;
    // alert(waveurl);
    var fileLen = waveurl.lastIndexOf(".");
    var fileName = waveurl.substring(0, fileLen);
    var picpath = fileName + '.jpg';
    // alert(picpath);
    document.getElementById('consignerRdSign').src = loadinggif;
    $.get('/uploadsth', { 'method': 'GET', 'data': waveurl }, function success() {
        imgpath1 = "/static/images/" + picpath;
        wavepath1 = "/static/audio/" + waveurl;
        document.getElementById('consignerRdSign').src = imgpath1;
        var audio = document.getElementById('wave1');
        audio.src = wavepath1;
        audio.load();
        // alert(imgpath1);
    });// send the file name to the upload
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

function Play1() {
    Play("wave1");
}

function Play2() {
    Play("wave2");
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