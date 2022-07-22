import React, {useState, useEffect} from 'react';
import './App.css';

const colorTable = [
  'silver',
  'gray',
  'green',
  'blue',
  'red',
  'fuchsia',
  'purple',
]

const classTable = [
  'No call',
  'Unknown',
  'Full pulse',
  'Sub-pulse transitory element',
  'Pulse body',
  'Bubble sub-pulse',
  'Grumble sub-pulse',
]

function App() {
  const [data, setData] = useState(null);

  useEffect(() => {
    fetch('data.json')
    .then(res => res.json())
    .then(json => {
      setData(json);
    })
  },[]);

  if (!data) {
    return <div/>
  }

  const {dev, test} = data;

  return (
    <div className="App">
      <h1>
        Visulization of Recurrent Sequence Modeling on Long Calls
      </h1>

      <h3>Colors</h3>

      <ul>
        {colorTable.map((value, index) =>
          <li 
            key={index}
            style={{color: value}}
          >
            {classTable[index]}
          </li>
        )}
      </ul>

      <div>
        <h3>
          Test on File <i>4T10lcFugit.wav</i>
        </h3>

        <audio controls className='Player'>
          <source 
            src="4T10lcFugit.wav" 
            type="audio/mpeg" 
          />
          Your browser does not support the audio element.
        </audio>

        <h3>Target ↓</h3>

        <div className='Strip-container'>
          <div className='Strip'>
            {test.target.map((value, index) =>
              <div 
                key={index}
                style={{background: colorTable[value]}}
                className='Strip-segment'
              />
            )}
          </div>
          
          <div className='Strip' style={{marginTop: '2px'}}>
            {test.pred.map((value, index) =>
              <div 
                key={index}
                style={{background: colorTable[value]}}
                className='Strip-segment'
              />
            )}
          </div>

          <div className='Strip' style={{paddingBottom: '10px'}}>
            {test.target.map((value, index) => 
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

      <div>
        <h3>
          Dev on File <i>5T10lcFugit.wav</i>
        </h3>

        <audio controls className='Player'>
          <source 
            src="5T10lcFugit.wav" 
            type="audio/mpeg" 
          />
          Your browser does not support the audio element.
        </audio>

        <h3>Target ↓</h3>

        <div className='Strip-container'>
          <div className='Strip'>
            {dev.target.map((value, index) =>
              <div 
                key={index}
                style={{background: colorTable[value]}}
                className='Strip-segment'
              />
            )}
          </div>
          
          <div className='Strip' style={{marginTop: '2px'}}>
            {dev.pred.map((value, index) =>
              <div 
                key={index}
                style={{background: colorTable[value]}}
                className='Strip-segment'
              />
            )}
          </div>

          <div className='Strip' style={{paddingBottom: '10px'}}>
            {dev.target.map((value, index) => 
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
