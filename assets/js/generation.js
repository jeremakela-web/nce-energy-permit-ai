/* ==========================================================
   NCE ENERGY — GENERATION
   ========================================================== */
// ==========================================================
// GENERATION
// ==========================================================

const GENERATION_API = 'https://ai.ncenergy.fi';

let generationStartedAt = 0;
let generationTimer = null;
let generationPoll = null;
let generationJobId = null;


// ----------------------------------------------------------
// STEP CONFIG
// ----------------------------------------------------------

const GENERATION_STEPS = {
    retrieval: 2,
    draft: 3,
    proofread: 4,
    raqs_final: 5,
    finalizing: 5,
    complete: 6
};



// ----------------------------------------------------------
// LIVE ACTIVITY
// ----------------------------------------------------------

function updateLiveActivity(data) {

    const container =
        document.getElementById('liveActivity');

    if (!container)
        return;


    let title = 'Processing...';
    let detail = 'Working on your permit application';


    if (data.stage === 'retrieval') {
        title = 'Retrieving regulatory sources...';
    
    } else if (data.stage === 'draft') {
        title = 'Generating regulatory draft...';
    
    } else if (data.stage === 'proofread') {
        title = 'Proofreading regulatory draft...';
    
    } else if (data.stage === 'raqs_final') {
        title = 'Running final quality assessment...';
    
    } else if (data.stage === 'finalizing') {
        title = 'Building final PDF...';
    
    } else if (data.stage === 'complete') {
        title = 'Permit application ready.';
    }


    let html = `
        <div class="live-item">
            <span class="live-dot"></span>
            <span>${title}</span>
            <small>NOW</small>
        </div>
    `;


    if (data.debug_sections) {

        const sections = data.debug_sections;

        html += `
            <div class="live-item">
                <span class="live-dot"></span>
                <span>
                    Processing document sections
                </span>
                <small>
                    ${Object.keys(sections).length} sections
                </small>
            </div>
        `;

    }


    if (data.stage === 'proofread') {

        html += `
            <div class="live-item">
                <span class="live-dot"></span>
                <span>
                    Checking generated text
                </span>
                <small>ACTIVE</small>
            </div>
        `;

    }


    if (data.stage === 'ready') {

        html += `
            <div class="live-item">
                <span class="live-dot"></span>
                <span>
                    Document generation completed
                </span>
                <small>DONE</small>
            </div>
        `;

    }


    container.innerHTML = html;
}

// ----------------------------------------------------------
// STEP UI
// ----------------------------------------------------------

// ----------------------------------------------------------
// STEP UI
// ----------------------------------------------------------

function setGenerationStep(step, state) {

    document
        .querySelectorAll('.generation-step')
        .forEach(el => {

            const n =
                Number(el.dataset.step);

            const status =
                el.querySelector('span');

            el.classList.remove(
                'active',
                'done'
            );

            if (status) {

                status.classList.remove(
                    'step-status-active'
                );

            }


            // ----------------------------------------------
            // COMPLETED STEPS
            // ----------------------------------------------

            if (n < step) {

                el.classList.add('done');

                if (status) {

                    status.textContent =
                        'COMPLETE';

                }

            }


            // ----------------------------------------------
            // CURRENT STEP
            // ----------------------------------------------

            if (n === step) {

                if (state === 'active') {

                    el.classList.add('active');

                    if (status) {

                        status.textContent =
                            'IN PROGRESS';

                        status.classList.add(
                            'step-status-active'
                        );

                    }

                }


                else if (state === 'done') {

                    el.classList.add('done');

                    if (status) {

                        status.textContent =
                            'COMPLETE';

                    }

                }

            }

        });

}


// ----------------------------------------------------------
// RESET STEPS
// ----------------------------------------------------------

function resetGenerationSteps() {

    document
        .querySelectorAll('.generation-step')
        .forEach(el => {

            el.classList.remove(
                'active',
                'done'
            );

            const status =
                el.querySelector('span');

            if (status)
                status.textContent = 'WAITING';

        });

}


// ----------------------------------------------------------
// TIMER
// ----------------------------------------------------------

function startElapsedTimer() {

    generationStartedAt =
        Date.now();

    clearInterval(
        generationTimer
    );


    generationTimer =
        setInterval(() => {

            const seconds =
                Math.floor(
                    (Date.now() -
                     generationStartedAt) / 1000
                );


            const hours =
                String(
                    Math.floor(seconds / 3600)
                ).padStart(2, '0');


            const minutes =
                String(
                    Math.floor(
                        (seconds % 3600) / 60
                    )
                ).padStart(2, '0');


            const secs =
                String(
                    seconds % 60
                ).padStart(2, '0');


            const el =
                document.getElementById(
                    'generationElapsed'
                );


            if (el) {

                el.textContent =
                    `${hours}:${minutes}:${secs}`;

            }

        }, 1000);

}


function stopElapsedTimer() {

    clearInterval(
        generationTimer
    );

    generationTimer = null;

}


// ----------------------------------------------------------
// STATUS TEXT
// ----------------------------------------------------------

function setGenerationStatus(text) {

    const el =
        document.getElementById(
            'generationStatus'
        );

    if (el)
        el.textContent = text;

}


// ----------------------------------------------------------
// DEBUG SECTIONS
// ----------------------------------------------------------

function showDebugSections(sections) {

    if (!sections)
        return;


    const values =
        Object.entries(sections);


    if (!values.length)
        return;


    // let html = `
    //     <div class="generation-debug">
    //         <strong>PROCESS DATA</strong>
    //         <div>
    // `;


    // values.forEach(([name, value]) => {

    //     html += `
    //         <span>
    //             ${name}: ${value}
    //         </span>
    //     `;

    // });


    // html += `
    //         </div>
    //     </div>
    // `;


    // const status =
    //     document.getElementById(
    //         'generationStatus'
    //     );


    // if (status)
    //     status.insertAdjacentHTML(
    //         'afterend',
    //         html
    //     );

}


// ----------------------------------------------------------
// REAL SERVER STAGE -> UI STEP
// ----------------------------------------------------------

function applyServerStage(data) {

    const stage =
        data.stage || 'retrieval';


    console.log(
        'GENERATION STAGE:',
        stage,
        data
    );


    const step =
        GENERATION_STEPS[stage] || 2;


    // Complete previous steps
    for (let i = 1; i < step; i++) {

        setGenerationStep(
            i,
            'done'
        );

    }


    // Current step
    if (stage === 'complete') {

        setGenerationStep(
            6,
            'done'
        );

        return;

    }


    setGenerationStep(
        step,
        'active'
    );


    // ------------------------------------------------------
    // HUMAN STATUS
    // ------------------------------------------------------

    const messages = {

        retrieval:
            'Retrieving regulatory context...',

        draft:
            'Generating regulatory draft...',

        proofread:
            'Proofreading generated document...',

        raqs_final:
            'Running final quality assessment...',

        finalizing:
            'Building final PDF...',

        complete:
            'Permit application ready.'

    };


    setGenerationStatus(
        messages[stage] ||
        'Processing application...'
    );


    // ------------------------------------------------------
    // REAL DEBUG DATA
    // ------------------------------------------------------

    if (data.debug_sections) {

        showDebugSections(
            data.debug_sections
        );

    }


    // ------------------------------------------------------
    // PHASE STATUS
    // ------------------------------------------------------

    if (data.phase_status) {

        console.log(
            'PHASE STATUS:',
            data.phase_status
        );

    }

}


// ----------------------------------------------------------
// FINAL RESULT
// ----------------------------------------------------------

function showGenerationComplete(data) {

    stopElapsedTimer();

    clearInterval(
        generationPoll
    );

    generationPoll = null;


    // Все шесть шагов завершены

    setGenerationStep(
        6,
        'done'
    );


    setGenerationStatus(
        'Permit application ready.'
    );


    // ------------------------------------------------------
    // PDF LINK
    // ------------------------------------------------------

    const downloadPdf =
        document.getElementById('downloadPdf');

    if (downloadPdf) {

        downloadPdf.href =
            `${GENERATION_API}/api/proofread/${generationJobId}/download`;

    }


    // ------------------------------------------------------
    // RESULT DATA
    // ------------------------------------------------------

    const resultJobId =
        document.getElementById('resultJobId');

    if (resultJobId) {

        resultJobId.textContent =
            generationJobId;

    }


    // ------------------------------------------------------
    // SHOW FINAL BLOCKS
    // ------------------------------------------------------
document.getElementById('resultsRow').style.display = 'grid';

document.getElementById('resultsRow').scrollIntoView({
    behavior: 'smooth',
    block: 'start'
});
    // const raqsSection =
    //     document.getElementById('raqsSection');

    // const resultSection =
    //     document.getElementById('resultSection');


    // if (raqsSection) {

    //     raqsSection.style.display =
    //         'block';

    // }


    // if (resultSection) {

    //     resultSection.style.display =
    //         'block';

    // }


    // ------------------------------------------------------
    // SCROLL TO RESULT
    // ------------------------------------------------------

    // if (raqsSection) {

    //     raqsSection.scrollIntoView({
    //         behavior: 'smooth',
    //         block: 'start'
    //     });

    // }

}

// ----------------------------------------------------------
// POLLING
// ----------------------------------------------------------

async function pollGeneration() {

    if (!generationJobId)
        return;


    try {

        const response =
            await fetch(
                `${GENERATION_API}/api/proofread/${generationJobId}`,
                {
                    cache: 'no-store'
                }
            );


        const data =
            await response.json();


        console.log(
            'GENERATION STATUS:',
            data
        );

        updateLiveActivity(data);


if (!response.ok) {

    if (
        response.status === 422 &&
        data.detail?.error === 'insufficient_sources'
    ) {

        clearInterval(generationPoll);
        stopElapsedTimer();

        setGenerationStatus(
            'Not enough regulatory sources for this project type.'
        );

        // закрываем генератор
        document.getElementById('generationPanel').style.display = 'none';

        // показываем нормальное сообщение
        alert(
            data.detail?.message ||
            'Not enough regulatory sources for this project type.'
        );

        return;
    }

    throw new Error(
        data.error ||
        data.detail?.message ||
        `HTTP ${response.status}`
    );
}


        // ----------------------------------------------
        // ERROR
        // ----------------------------------------------

        if (
            data.status === 'error' ||
            data.error
        ) {

            clearInterval(
                generationPoll
            );

            stopElapsedTimer();


            setGenerationStatus(
                'Generation failed: ' +
                (data.error || 'Unknown error')
            );


            return;

        }


        // ----------------------------------------------
        // SERVER STAGE
        // ----------------------------------------------

        applyServerStage(
            data
        );


        // ----------------------------------------------
        // COMPLETE
        // ----------------------------------------------

        if (
            data.status === 'done' ||
            data.stage === 'complete'
        ) {
             updateRAQS(data.raqs);
         
    document.getElementById('resultGenerated')
        .textContent =
            `Generated: ${
                data.raqs?.created_at
                    ? new Date(data.raqs.created_at).toLocaleString()
                    : new Date().toLocaleString()
            }`;

            showGenerationComplete(
                data
            );

        }

    }
    catch (error) {

        console.error(
            'POLL ERROR:',
            error
        );

        // Don't kill generation because
        // one polling request failed.

        setGenerationStatus(
            'Waiting for server...'
        );

    }

}


// ----------------------------------------------------------
// START POLLING
// ----------------------------------------------------------

function startGenerationPolling() {

    clearInterval(
        generationPoll
    );


    // first request immediately

    pollGeneration();


    // then every 7 seconds

    generationPoll =
        setInterval(
            pollGeneration,
            7000
        );

}


// ----------------------------------------------------------
// FORM VALUE
// ----------------------------------------------------------

function fieldValue(id) {

    const el =
        document.getElementById(id);

    return el
        ? el.value.trim()
        : '';

}




function updateResultSummary(payload) {

    document.getElementById('summaryType').textContent =
        payload.hanketyyppi || '—';

    document.getElementById('summaryApplicant').textContent =
        payload.hakija || '—';

    document.getElementById('summaryBusinessId').textContent =
        payload.y_tunnus || '—';

    document.getElementById('summaryProperty').textContent =
        payload.kiinteistotunnus || '—';

    document.getElementById('summaryMunicipality').textContent =
        payload.kunta || '—';

    document.getElementById('summaryPower').textContent =
        payload.teho_mw ? payload.teho_mw + ' MW' : '—';

    document.getElementById('summaryPhase').textContent =
        payload.hankkeen_vaihe || '—';

    document.getElementById('summaryAuthority').textContent =
        payload.kohdeviranomainen || '—';

    document.getElementById('resultProject').textContent =
        `${payload.hanketyyppi || 'Project'} · ${payload.country || ''} · ${payload.teho_mw || '—'} MW`;
}





// ----------------------------------------------------------
// GENERATE
// ----------------------------------------------------------

async function generateApplication() {

    const panel =
        document.getElementById(
            'generationPanel'
        );


    if (!panel)
        return;


    panel.style.display =
        'block';


    resetGenerationSteps();

    startElapsedTimer();


    setGenerationStep(
        1,
        'active'
    );


    setGenerationStatus(
        'Sending project to AI pipeline...'
    );


    // ------------------------------------------------------
    // PAYLOAD
    // ------------------------------------------------------


// const payload = {
//     hanketyyppi: 'tuulivoima_maa',
//     hakija: 'Serkov',
//     y_tunnus: '1234567-8',
//     osoite: 'Test',
//     kiinteistotunnus: '019-1001-1000',
//     kunta: 'Helsinki',
//     teho_mw: 11,
//     sijainti_ymparistovaikutukset: 'Private area',
//     hankkeen_vaihe: 'Esiselvitys',
//     kohdeviranomainen: 'Kunta (rakentamislupa)',
//     lang: 'EN',
//     country: 'FI',
//     session_id: 'sess_' + Math.random().toString(36).substring(2, 18),
//     hanke_id: '1234567_8__019_1001_1000'
// };





// ------------------------------------------------------
// PAYLOAD
// ------------------------------------------------------

const PROJECT_TYPES = {

    wind: 'tuulivoima_maa',
    windsea: 'offshore_wind',
    solar: 'aurinkovoima',
    bess: 'BESS',
    smr: 'SMR',

    hydro: 'vesivoima',
    residential: 'asuinrakennus',
    industrial: 'teollisuus',
    agriculture: 'maatalous',
    commercial: 'liikerakennus',
    datacenter: 'datakeskus',
    hybrid: 'hybridi',
    other: 'muu'

};


const payload = {

    hanketyyppi:
        PROJECT_TYPES[
            document.getElementById('hanke_tyyppi').value
        ],

    kiinteistotunnus:
        document.getElementById('kiinteistotunnus').value,

    kunta:
        document.getElementById('kunta').value,

    hakija:
        document.getElementById('hakija').value,

    teho_mw:
        Number(
            document.getElementById('teho_mw').value
        ) || 0,

    kapasiteetti_mwh:
        Number(
            document.getElementById('kapasiteetti_mwh').value
        ) || 0,

    y_tunnus:
        document.getElementById('y_tunnus').value,

    osoite:
        document.getElementById('osoite').value,

    sijainti_ymparistovaikutukset:
        document.getElementById(
            'sijainti_ymparistovaikutukset'
        ).value,

    hankkeen_vaihe:
        document.getElementById('hankkeen_vaihe').value,

    kohdeviranomainen:
        document.getElementById('kohdeviranomainen').value,

    // lang:
    //     currentCountry === 'EE'
    //         ? 'ET'
    //         : 'EN',

    lang:
        {
            FI: 'FI',
            SE: 'SV',
            DA: 'DA',
            NO: 'NO',
            PL: 'PL',
            DE: 'DE',
            EE: 'ET',
            LV: 'LV',
            LT: 'LT'
        }[currentCountry] || 'EN',

    country:
        currentCountry,

    session_id:
        'sess_' +
        Math.random()
            .toString(36)
            .substring(2, 18),

    hanke_id:
        document.getElementById('hanke_id').value

};


console.log(
    'PRODUCTION PAYLOAD:',
    payload
);

updateResultSummary(payload);




// const payload = {

//     // Тип проекта
//     // Project type
//     // tuulivoima_maa = наземная ветроэнергетика / onshore wind
//     hanketyyppi: 'tuulivoima_maa',


//     // Идентификатор недвижимости / земельного участка
//     // Property ID / cadastral property identifier
//     kiinteistotunnus: '636-439-4-711',


//     // Муниципалитет
//     // Municipality
//     kunta: 'Pöytyä',


//     // Заявитель / компания или физическое лицо
//     // Applicant
//     hakija: 'Acme Energia Oy',


//     // Общая установленная мощность проекта, МВт
//     // Total project power / installed capacity in MW
//     teho_mw: 42.0,


//     // Энергетическая ёмкость, МВт·ч
//     // Energy capacity in MWh
//     // Для ветра обычно 0 или не используется
//     kapasiteetti_mwh: 0.0,


//     // Business ID / идентификационный номер компании
//     // Finnish Business ID (Y-tunnus)
//     y_tunnus: '1234567-8',


//     // Адрес проекта / заявителя
//     // Address
//     osoite: '',


//     // Краткое описание расположения и экологического воздействия
//     // Location / environmental impacts
//     sijainti_ymparistovaikutukset: '',


//     // Стадия проекта
//     // Project phase
//     // esiselvitys = предварительное исследование / pre-study
//     // lupavaihe = разрешительная стадия / permitting
//     // rakentaminen = строительство / construction
//     hankkeen_vaihe: 'Esiselvitys',


//     // Целевой орган власти
//     // Target authority
//     kohdeviranomainen: '',


//     // Язык генерируемого документа
//     // Output language
//     lang: 'EN',


//     // Страна проекта
//     // Project country
//     country: 'FI',


//     // ID пользовательской сессии
//     // Session ID
//     session_id:
//         'sess_' + Math.random().toString(36).substring(2, 18),


//     // ID проекта
//     // Project ID
//     // Может быть пустым, если проект ещё не имеет ID
//     hanke_id: ''

// };

// updateResultSummary(payload);



    try {

        // --------------------------------------------------
        // POST
        // --------------------------------------------------

        const response =
            await fetch(
                `${GENERATION_API}/api/generate-application`,
                {
                    method: 'POST',

                    headers: {
                        'Content-Type':
                            'application/json'
                    },

                    body:
                        JSON.stringify(payload)
                }
            );


        const data =
            await response.json();


        console.log(
            'GENERATE RESPONSE:',
            data
        );


        if (!response.ok) {

            throw new Error(
                data.error ||
                `HTTP ${response.status}`
            );

        }


        if (!data.job_id) {

            throw new Error(
                'Server did not return job_id'
            );

        }


        // --------------------------------------------------
        // JOB
        // --------------------------------------------------

        generationJobId =
            data.job_id;


        document
            .getElementById(
                'generationJobId'
            )
            .textContent =
            generationJobId;

document.getElementById('resultProject').textContent =
    `${payload.hanketyyppi} · ${payload.country} · ${payload.teho_mw} MW`;
        // POST accepted

        setGenerationStep(
            1,
            'done'
        );


        setGenerationStep(
            2,
            'active'
        );


        setGenerationStatus(
            'Job created. Retrieving regulatory context...'
        );


        // --------------------------------------------------
        // START REAL POLLING
        // --------------------------------------------------

        startGenerationPolling();

    }
    catch (error) {

        console.error(
            'GENERATION ERROR:',
            error
        );


        stopElapsedTimer();

        clearInterval(
            generationPoll
        );


        setGenerationStatus(
            'Generation failed: ' +
            error.message
        );

    }

}

// ----------------------------------------------------------
// RAQS — заполнение реальными данными сервера
// ----------------------------------------------------------

function updateRAQS(raqs) {

    if (!raqs || !raqs.scores)
        return;


    // Общая оценка приходит от сервера по шкале 1–5.
    // UI показывает шкалу 0–100.
    const overall =
        Math.round((raqs.overall || 0) * 20);


    document.getElementById('raqsScore')
        .textContent = overall;


    const scores = raqs.scores;


    const citations =
        scores.viittaukset?.pisteet || 0;

    const coverage =
        scores.lupakattavuus?.pisteet || 0;

    const uncertainty =
        scores.epävarmuus?.pisteet || 0;

    const comprehensiveness =
        scores.kattavuus?.pisteet || 0;

    const readiness =
        scores.valmisteluaste?.pisteet || 0;


    // Заполняем полосы и цифры.
    // Сервер: 1–5 → UI: 20–100.
    document.getElementById('raqsCitations')
        .style.width = `${citations * 20}%`;

    document.getElementById('raqsCitationsValue')
        .textContent = citations * 20;


    document.getElementById('raqsCoverage')
        .style.width = `${coverage * 20}%`;

    document.getElementById('raqsCoverageValue')
        .textContent = coverage * 20;


    document.getElementById('raqsUncertainty')
        .style.width = `${uncertainty * 20}%`;

    document.getElementById('raqsUncertaintyValue')
        .textContent = uncertainty * 20;


    document.getElementById('raqsComprehensiveness')
        .style.width = `${comprehensiveness * 20}%`;

    document.getElementById('raqsComprehensivenessValue')
        .textContent = comprehensiveness * 20;


    document.getElementById('raqsReadiness')
        .style.width = `${readiness * 20}%`;

    document.getElementById('raqsReadinessValue')
        .textContent = readiness * 20;


    // ------------------------------------------------------
    // Комментарии / flagged от сервера
    // ------------------------------------------------------
    // ------------------------------------------------------
    // RAQS — пояснения от сервера
    // ------------------------------------------------------
    
    let details =
        document.getElementById('raqsDetails');
    
    if (!details) {
    
        details = document.createElement('div');
    
        details.id = 'raqsDetails';
        details.className = 'raqs-details';
    
        document
            .getElementById('raqsWarning')
            .after(details);
    }
    
    details.innerHTML = '';
    
    const labels = {
        viittaukset: 'Citations',
        lupakattavuus: 'Permit Coverage',
        epävarmuus: 'Uncertainty',
        kattavuus: 'Comprehensiveness',
        valmisteluaste: 'Readiness'
    };
    
    Object.entries(scores).forEach(
        ([key, value]) => {
    
            if (!value?.perustelu)
                return;
    
            const block =
                document.createElement('div');
    
            block.className =
                'raqs-detail';
    
            block.innerHTML = `
                <strong>
                    ${labels[key] || key}
                    · ${value.pisteet}/5
                </strong>
    
                <p>
                    ${value.perustelu}
                </p>
            `;
    
            details.appendChild(block);
        }
    );
    // const warning =
    //     document.getElementById('raqsWarning');

    // const warningText =
    //     document.getElementById('raqsWarningText');


    // if (raqs.flagged && raqs.flagged.length) {

    //     warning.style.display = 'flex';

    //     warningText.textContent =
    //         `${raqs.flagged.length} requirements need expert verification`;

    // } else {

    //     warning.style.display = 'none';

    // }

}




// ----------------------------------------------------------
// FORM SUBMIT
// ----------------------------------------------------------






document
    .getElementById('permitForm')
    .addEventListener(
        'submit',
        function(e) {

            e.preventDefault();



            document.getElementById('generationPanel').style.display = 'block';
            
            // document.getElementById('raqsSection').style.display = 'none';
            // document.getElementById('resultSection').style.display = 'none';
            
            document
                .getElementById('generationPanel')
                .scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });


            generateApplication();

        }
    );





// document.getElementById('raqsScore').textContent = '87';

// document.getElementById('raqsCitations').style.width = '92%';
// document.getElementById('raqsCitationsValue').textContent = '92';

// document.getElementById('raqsCoverage').style.width = '84%';
// document.getElementById('raqsCoverageValue').textContent = '84';

// document.getElementById('raqsUncertainty').style.width = '91%';
// document.getElementById('raqsUncertaintyValue').textContent = '91';

// document.getElementById('raqsComprehensiveness').style.width = '83%';
// document.getElementById('raqsComprehensivenessValue').textContent = '83';

// document.getElementById('raqsReadiness').style.width = '88%';
// document.getElementById('raqsReadinessValue').textContent = '88';


