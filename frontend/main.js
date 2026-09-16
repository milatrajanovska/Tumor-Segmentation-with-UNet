const uploadBox=document.getElementById('uploadBox');
const mriInput=document.getElementById('mriInput');
const fileName=document.getElementById('fileName');
const start=document.getElementById('start');
let polling;
if(start) {
    var slice_num = 100;
    const slider = document.getElementById('myRange');
    const showSliceNum = document.getElementById('show_slice_num');
    var file;
    slider.oninput = function () {
        slice_num = slider.value;
        localStorage.setItem("sliceNumber",slice_num);
        showSliceNum.innerHTML = slice_num;
    }
    mriInput.addEventListener("change", function () {
        fileName.style.display = "block";
        if (mriInput.files.length > 0) {
             file = mriInput.files[0];
            fileName.textContent = file.name;
            localStorage.setItem("fileName",file.name)
        }
    })

    uploadBox.addEventListener("dragover", function (event) {
        event.preventDefault();
    })
    uploadBox.addEventListener("drop", function (event) {
        event.preventDefault()
        fileName.style.display = "block";

        file = event.dataTransfer.files[0];
        if (file) {
            fileName.textContent = file.name;
            localStorage.setItem("fileName",file.name)

        }
    })

    start.addEventListener('click', function () {
        formData = new FormData();
        formData.append("file", file);
        formData.append("slice_number", slice_num)

        fetch('http://localhost:8000/upload', {
            method: 'POST',
            body: formData
        }).then(res => res.json())
            .then(data => {
                console.log(data);
                if(!data.job_id){
                    alert("Serverot ne vrati job id")
                    return;
                }
                localStorage.setItem("job_id", data.job_id)
                window.location.href = "processing.html"
            })
            .catch(err => console.log(err));
    })
}


const jobId=localStorage.getItem("job_id");

if(window.location.pathname.includes("processing.html")) {
    if (jobId && jobId !== "undefined" && jobId !== "null") {
        checkStatus(jobId)
        polling = setInterval(() => checkStatus(jobId), 5000);
    } else {
        console.log("That job is not found")
    }
}

function checkStatus(jobId){
    fetch('http://localhost:8000/status/'+jobId,{
        method: 'GET',
    }).then(res => res.json())
        .then(data => {
            console.log(data);
            updateUI(data.steps)
            const allDone = Object.values(data.steps).every(s => s === "done");
            if (allDone || data.error) {
                clearInterval(polling); // запри полирање
                if (data.error) {
                    console.error("Job error:", data.error);
                } else {
                    console.log("Job completed!");
                    localStorage.setItem("mask_path", data.result_files.mask);
                    localStorage.setItem("flair_path", data.result_files.flair);
                    localStorage.setItem("combination_path", data.result_files.combination);
                    localStorage.setItem("IsTumor",data.result_files.IsTumor);
                    localStorage.setItem("percent",data.result_files.percent);

                    // тука можеш да пренасочиш кон results страница
                    window.location.href = "preview.html";
                }
            }
        })
        .catch(err => console.log(err));
}

function updateUI(steps){
    for(const step in steps){
        updateStepUI(step,steps[step]);
    }
}

function updateStepUI(step,status){
    const svgElement=document.getElementById("circle_"+step);
    console.log("Status",status)
    if(status=="done"){
        svgElement.outerHTML = `
            <svg id="circle_${step}" xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 48 48">
                <linearGradient id="grad1_${step}" x1="21.241" x2="3.541" y1="39.241" y2="21.541" gradientUnits="userSpaceOnUse"><stop offset=".108" stop-color="#0d7044"></stop><stop offset=".433" stop-color="#11945a"></stop></linearGradient>
                <path fill="url(#grad1_${step})" d="M16.599,41.42L1.58,26.401c-0.774-0.774-0.774-2.028,0-2.802l4.019-4.019c0.774-0.774,2.028-0.774,2.802,0L23.42,34.599c0.774,0.774,0.774,2.028,0,2.802l-4.019,4.019C18.627,42.193,17.373,42.193,16.599,41.42z"></path>
                <linearGradient id="grad2_${step}" x1="-15.77" x2="26.403" y1="43.228" y2="43.228" gradientTransform="rotate(134.999 21.287 38.873)" gradientUnits="userSpaceOnUse"><stop offset="0" stop-color="#2ac782"></stop><stop offset="1" stop-color="#21b876"></stop></linearGradient>
                <path fill="url(#grad2_${step})" d="M12.58,34.599L39.599,7.58c0.774-0.774,2.028-0.774,2.802,0l4.019,4.019c0.774,0.774,0.774,2.028,0,2.802L19.401,41.42c-0.774,0.774-2.028,0.774-2.802,0l-4.019-4.019C11.807,36.627,11.807,35.373,12.58,34.599z"></path>
            </svg>`;
    }else{
        svgElement.outerHTML = `
            <svg id="circle_${step}" width="20" height="20">
                <circle cx="10" cy="10" r="8" stroke="white" stroke-width="2" fill="none" />
            </svg>`;
    }
}


if (window.location.pathname.includes("preview.html")) {
    const maskdiv = document.getElementById("mask");
    const flairdiv = document.getElementById("flair");
    const combinationDiv = document.getElementById("combination");
    const sliceParagraph = document.getElementById("slice");
    const imageParagraph = document.getElementById("image");
    const detected_tumor = document.getElementById("detected_tumor");
    const tumor_area = document.getElementById("tumor_area");

    const savedMask = localStorage.getItem("mask_path");
    const savedFlair = localStorage.getItem("flair_path");
    const savedCombination = localStorage.getItem("combination_path");

    const isTumor = localStorage.getItem("IsTumor");
    const percent = localStorage.getItem("percent");
    const sliceNum = localStorage.getItem("sliceNumber");
    console.log(sliceNum);
    const flair_image_name = localStorage.getItem("fileName");

    // Приказ на сликите
    if (maskdiv && savedMask) {
        maskdiv.innerHTML = `<img src="${savedMask}" style="max-width:100%;">`;
    }
    if (flairdiv && savedFlair) {
        flairdiv.innerHTML = `<img src="${savedFlair}" style="max-width:100%;">`;
    }
    if (combinationDiv && savedCombination) {
        combinationDiv.innerHTML = `<img src="${savedCombination}" style="max-width:100%;">`;
    }

    // Поставување на текстовите
    if (sliceParagraph && sliceNum) {
        sliceParagraph.innerText = `${sliceNum}/155`;
    }
    if (imageParagraph && flair_image_name) {
        imageParagraph.innerText = flair_image_name;
    }
    if (detected_tumor && isTumor !== null) {
        detected_tumor.innerHTML = (isTumor === "true" || isTumor === true) ? "YES" : "NO";
    }
    if (tumor_area && percent !== null) {
        tumor_area.innerHTML = `${percent}%`;
    }

    // Копче за нова слика
    const newImageButton = document.getElementById("new_image");
    if (newImageButton) {
        newImageButton.addEventListener("click", function() {
            window.location.href = "input.html";
        });
    }


}
