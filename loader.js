// loader.js

const chronobarBox = document.getElementById('chronobar');
chronobarBox.date = chronobarBox.querySelector('.date');
chronobarBox.pics = chronobarBox.querySelector('.pics');
chronobarBox.bar = chronobarBox.querySelector('.time-bar');
chronobarBox.bar.track = chronobarBox.bar.querySelector('.time-track');


// parsing objs vars
let dates = [];
let picObjs = [];
piclist.forEach((picname, idx) => {
  let picinfo = picname.split('_');
  if (picinfo.length != 2) return;
  let dateObj= new Date(picinfo[0]);
  if (isNaN(dateObj)) return;
  if (dates.includes(picinfo[0])) return;

  dates.push(picinfo[0]);
  let picPath = pathToFiles + picname ;

  // img creation
  let pic = document.createElement('img');
  pic.src = picPath; pic.picAuthor = picinfo[1];
  pic.picDate = dateObj; pic.picDateStr = picinfo[0];
  if(idx != 0) pic.classList.add('disabled');
  picObjs.push(pic);
  //chronobar.pics.appendChild(pic);
});


// sort picObjs, create marks && append to chronobar
let marks = [];
picObjs.sort((a, b) => a.picDate - b.picDate);
chronobarBox.date.innerText = picObjs[0].picDateStr;
{
  let init = picObjs[0].picDate,
      end = picObjs[picObjs.length - 1].picDate;
  let oneDay = 24 * 60 * 60 * 1000;
  let days = Math.round((end - init)/oneDay);
  picObjs.forEach((obj) => {
    let mark = document.createElement('mark');
    mark.classList.add('time-mark');
    mark.left = ((obj.picDate - init)/oneDay)/days*100;
    mark.style.left = mark.left + '%';
    obj.mark = mark;
    mark.pic = obj;
    marks.push(mark);
    chronobarBox.bar.appendChild(mark);
    chronobar.pics.appendChild(obj);
  });
}


// bttn event
var runningEv = null;
function togglePower() {
  if (runningEv) {
    clearInterval(runningEv);
    runningEv = null;
  } else {
    runningEv = setInterval(() => {
      let active = chronobarBox.pics.querySelector(':not(.disabled)');
      let nextPic = active.nextElementSibling ?
                    active.nextElementSibling : active.parentElement.firstElementChild;
      nextPic.classList.toggle('disabled');
      active.classList.toggle('disabled');
      chronobarBox.bar.track.style.width = nextPic.mark.style.left;
      chronobarBox.date.innerText = nextPic.picDateStr;
    }, 1000);
  }
}

chronobarBox.querySelector('.control-bttn').onclick = function() {
  this.firstElementChild.classList.toggle('paused');
  togglePower();
}

// Time bar click
let isDragging = false;
function closestMark(percentage) {
  return marks.reduce((prev, curr) =>
    Math.abs(curr.left - percentage) < Math.abs(prev.left - percentage) ? curr : prev
  );
}

function changeTimeBar(newX = 0) {
  const box = chronobarBox.bar.getBoundingClientRect();
  const percentage = ((newX - box.left) / box.width ) * 100;
  mark = closestMark(percentage);
  // Display result
  chronobarBox.pics.querySelector(':not(.disabled)').classList.toggle('disabled');
  mark.pic.classList.toggle('disabled');
  chronobarBox.bar.track.style.width = mark.style.left;
  chronobarBox.date.innerText = mark.pic.picDateStr;
}

function onMouseDown(ev) { isDragging = true; ev.preventDefault();}
function onMouseMove(ev) {
  if (isDragging) {
    const clientX = ev.touches ? ev.touches[0].clientX : ev.clientX;
    changeTimeBar(clientX);
    ev.preventDefault();
  }
}
function onMouseUp(ev) {
  if (isDragging) {
    isDragging = false;
    const clientX = ev.touches ? ev.touches[0].clientX : ev.clientX;
    changeTimeBar(clientX);
  }
}

chronobarBox.bar.addEventListener('mousedown', onMouseDown);
document.addEventListener('mousemove', onMouseMove);
document.addEventListener('mouseup', onMouseUp);

chronobarBox.bar.addEventListener('touchstart', onMouseDown, { passive: false });
document.addEventListener('touchmove', onMouseMove, { passive: false });
document.addEventListener('touchend', onMouseUp);


//
