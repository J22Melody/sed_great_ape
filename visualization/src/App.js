import React, {useState, useEffect} from 'react';
import './App.css';

const colorTable = [
  'silver',
  // 'gray',
  'green',
  'blue',
  'red',
  'fuchsia',
  'purple',
  'orange',
  'brown',
]

// const classTable = [
//   'No call',
//   'Unknown',
//   'Full pulse',
//   'Sub-pulse transitory element',
//   'Pulse body',
//   'Bubble sub-pulse',
//   'Grumble sub-pulse',
// ]

function App() {
  const [pred, setPred] = useState([]);
  const [target, setTarget] = useState([]);

  const urlSearchParams = new URLSearchParams(window.location.search);
  const params = Object.fromEntries(urlSearchParams.entries()); 

  const dataset = params.dataset;
  const model = params.model;
  const filename = params.filename;

  const result_name = `${dataset}_${model}/${filename}`;

  useEffect(() => {
    const predUrl = `data/${result_name}.pred.txt`;
    const targetUrl = `data/${result_name}.target.txt`;

    fetch(predUrl)
    .then((response) => response.text())
    .then(data => {
      setPred(data.split('\n').map(Number));
    })

    fetch(targetUrl)
    .then((response) => response.text())
    .then(data => {
      setTarget(data.split('\n').map(Number));
    })
  },[result_name]);

  const audio_url = `data/${dataset}_raw/${filename}.WAV`

  return (
    <div className="App">
      <h1>
        Visulization of Recurrent Sequence Modeling on Calls
      </h1>

      <h3>Colors</h3>

      <ul>
        {colorTable.map((value, index) =>
          <li 
            key={index}
            style={{color: value}}
          >
            class {index}
          </li>
        )}
      </ul>

      <div>
        <h3>
          Test results on <i>{result_name}</i>
        </h3>

        <audio controls className='Player'>
          <source 
            src={audio_url}
            type="audio/mpeg" 
          />
          Your browser does not support the audio element.
        </audio>

        <h3>Target ↓</h3>

        <div className='Strip-container'>
          <div className='Strip'>
            {target.map((value, index) =>
              <div 
                key={index}
                style={{background: colorTable[value]}}
                className='Strip-segment'
              />
            )}
          </div>
          
          <div className='Strip' style={{marginTop: '2px'}}>
            {pred.map((value, index) =>
              <div 
                key={index}
                style={{background: colorTable[value]}}
                className='Strip-segment'
              />
            )}
          </div>

          <div className='Strip' style={{paddingBottom: '10px'}}>
            {target.map((value, index) => 
              <div
                key={index}
                className='Strip-axis'
              >
                {(index % 25 === 0) &&
                  <span>| {index / 50}</span>
                }
              </div>
            )}
          </div>
        </div>

        <h3 style={{marginTop: '4px'}}>Predicted ↑</h3>
      </div>
    </div>
  );
}

export default App;
