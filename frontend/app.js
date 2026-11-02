const $ = id => document.getElementById(id);
let config;
async function init(){
  try{
    const r=await fetch('/api/config'); if(!r.ok) throw new Error('API config unavailable');
    config=await r.json();
    const select=$('model'); select.innerHTML='';
    Object.entries(config.models).forEach(([key,m])=>{
      const o=document.createElement('option');o.value=key;o.textContent=`${m.family} · ${m.size_b}B · ${m.name}`;select.appendChild(o);
      const card=document.createElement('div');card.className='model-item';
      card.innerHTML=`<b>${m.family} ${m.size_b}B</b><small>${m.tier.toUpperCase()} · ${key}</small>`;
      $('modelList').appendChild(card);
    });
    $('modelCount').textContent=`${Object.keys(config.models).length} MODELS`;
    select.value=config.default_model;$('message').textContent='Backend connected. FP16 inference is ready.';
  }catch(e){$('message').textContent='Backend connection failed: '+e.message;}
}
async function poll(jobId){
  for(let i=0;i<600;i++){
    const r=await fetch(`/api/jobs/${jobId}`);const job=await r.json();
    $('jobStatus').textContent=job.status.toUpperCase();
    if(job.status==='completed'){
      const x=job.result.result;
      $('latency').textContent=x.latency_seconds;$('throughput').textContent=x.tokens_per_second;
      $('allocated').textContent=x.gpu_memory_allocated_gb;$('reserved').textContent=x.gpu_memory_reserved_gb;
      $('response').textContent=x.text || '(No new text generated)';
      $('message').textContent=`Completed on ${x.device}. Results saved to data/results.`;
      $('runBtn').disabled=false;return;
    }
    if(job.status==='failed'){
      $('response').textContent=job.error;$('message').textContent='Experiment failed. Check backend terminal.';
      $('runBtn').disabled=false;return;
    }
    await new Promise(resolve=>setTimeout(resolve,1000));
  }
  $('message').textContent='Polling timed out; check backend terminal.';$('runBtn').disabled=false;
}
$('runBtn').addEventListener('click',async()=>{
  const payload={model_key:$('model').value,precision:'FP16',prompt:$('prompt').value,max_new_tokens:Number($('tokens').value)};
  $('runBtn').disabled=true;$('jobStatus').textContent='SUBMITTING';$('message').textContent='Submitting experiment...';
  $('response').textContent='Running model inference...';
  try{
    const r=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const data=await r.json();if(!r.ok)throw new Error(data.detail||'Request failed');
    await poll(data.job_id);
  }catch(e){$('message').textContent=e.message;$('runBtn').disabled=false;$('jobStatus').textContent='ERROR';}
});
init();
