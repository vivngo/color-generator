
const HOSTED_URLS = {
  model:
      'model_js/model.json',
  metadata:
      'model_js/metadata.json'
};

const examples = {
  'example1':
      'blue green',
  'example2':
      'red cherry',
  'example3':
      'cherry blossom',
  'example4':
      'beautiful afternoon'
};


function status(statusText) {
  // console.log(statusText);
  document.getElementById('status').innerHTML = statusText;
}

function showMetadata(metadataJSON) {
  document.getElementById('vocabularySize').textContent =
      metadataJSON['vocabulary_size'];
  document.getElementById('maxLen').textContent =
      metadataJSON['max_len'];
}

function settextField(text, predict) {
  const textField = document.getElementById('text-entry');
  textField.value = text;
  doPredict(predict);
}

function setPredictFunction(predict) {
  const textField = document.getElementById('text-entry');
  textField.addEventListener('input', () => doPredict(predict));
}

function disableLoadModelButtons() {
  // document.getElementById('load-model').style.display = 'none';
  document.getElementById('load-model').disabled = true;
  document.getElementById('load-model').innerText = "Model loaded";
}

function doPredict(predict) {
  const textField = document.getElementById('text-entry');
  const result = predict(textField.value);
  rgbstring = "(" + result.rgb[0] + ", "
  rgbstring += result.rgb[1] + ", "
  rgbstring += result.rgb[2] + ")"
  outstring = "<strong>" + textField.value + ":</strong> "
  outstring += "<i>(time elapsed: " + result.elapsed.toFixed(3) + " ms)</i>"
  outstring += "<br> R G B: " + rgbstring
  status(outstring);

  console.log(rgbstring)
  var swatch = document.getElementById('color-swatch');
  swatch.style.background = "rgb" + rgbstring;
}

function prepUI(predict) {
  setPredictFunction(predict);
  const testExampleSelect = document.getElementById('example-select');
  testExampleSelect.addEventListener('change', () => {
    settextField(examples[testExampleSelect.value], predict);
  });
  settextField(examples['example1'], predict);
}

function normalize(rgb) {
  return rgb.map(function(x) { return Math.round(x * 255); });
}

async function urlExists(url) {
  status('Testing url ' + url);
  try {
    const response = await fetch(url, {method: 'HEAD'});
    return response.ok;
  } catch (err) {
    return false;
  }
}

async function loadHostedPretrainedModel(url) {
  status('Loading pretrained model from ' + url);
  try {
    const model = await tf.loadLayersModel(url);
    status('Done loading pretrained model.');
    disableLoadModelButtons();
    return model;
  } catch (err) {
    console.error(err);
    status('Loading pretrained model failed.');
  }
}

async function loadHostedMetadata(url) {
  status('Loading metadata from ' + url);
  try {
    const metadataJson = await fetch(url);
    const metadata = await metadataJson.json();
    status('Done loading metadata.');
    return metadata;
  } catch (err) {
    console.error(err);
    status('Loading metadata failed.');
  }
}

class Classifier {

  async init(urls) {
    this.urls = urls;
    this.model = await loadHostedPretrainedModel(urls.model);
    await this.loadMetadata();
    return this;
  }

  async loadMetadata() {
    const metadata =
        await loadHostedMetadata(this.urls.metadata);
    showMetadata(metadata);
    this.maxLen = metadata['max_len'];
    console.log('maxLen = ' + this.maxLen);
    this.wordIndex = metadata['word_index']
  }

  predict(text) {
    // Convert to lower case and remove all punctuations.
    const inputText =
        text.trim().toLowerCase().replace(/(\.|\,|\!)/g, '');
    // Look up word indices.
    const inputBuffer = tf.buffer([1, this.maxLen], 'float32');

    // Use offset to pre-pad instead of post-pad tensor
    const offset = Math.max(0, this.maxLen - inputText.length)
    for (let i = 0; i < inputText.length; ++i) {
      const word = inputText[i];
      inputBuffer.set(this.wordIndex[word], 0, offset+i);
      console.log(word, this.wordIndex[word], inputBuffer);
    }
    const input = inputBuffer.toTensor();
    console.log('input: ' + input);

    status('Running inference');
    const beginMs = performance.now();
    const predictOut = this.model.predict(input);
    console.log('predictOut: ' + predictOut.dataSync());
    const rgb = predictOut.dataSync();//[0];
    predictOut.dispose();
    const endMs = performance.now();

    return {rgb: normalize(rgb), elapsed: (endMs - beginMs)};
  }

};

async function setup() {
  if (await urlExists(HOSTED_URLS.model)) {
    status('Model available: ' + HOSTED_URLS.model);
    const button = document.getElementById('load-model');
    button.addEventListener('click', async () => {
      const predictor = await new Classifier().init(HOSTED_URLS);
      prepUI(x => predictor.predict(x));
    });
    button.style.display = 'inline-block';
  }

  status('Standing by...');
}

setup();
