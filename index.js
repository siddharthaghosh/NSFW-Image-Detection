require('dotenv').config();
const express = require('express');
const axios = require('axios');
const tf = require('@tensorflow/tfjs-node');
const nsfw = require('nsfwjs');
const app = express();

const DOMAIN = process.env.DOMAIN || 'localhost';
const PORT = process.env.PORT || 9832;
const AUTHORIZATION_KEY = process.env.AUTHORIZATION_KEY;

const NSFW_MODEL =
  process.env.NSFW_MODEL || 'https://ml.files-sashido.cloud/models/nsfw_mobilenet_v2/93/';
const NSFW_MODEL_SIZE = process.env.NSFW_MODEL_SIZE || 224;

let nsfwModel;

const authorize = (req, res, next) => {
  if (
    !req.headers?.authorization ||
    (AUTHORIZATION_KEY && req.headers.authorization != AUTHORIZATION_KEY)
  )
    return res.send('You are not authorized');
  next();
};

app.post('/nsfw', authorize, async (req, res) => {
  if (!req.query?.url) return res.send('You need to supply url in the query');
  let results = await classifyImage(req.query.url);
  res.json(results);
});

const classifyImage = async (url) => {
  let pic;
  let result = {};

  try {
    pic = await axios.get(url, { responseType: 'arraybuffer' });
  } catch (err) {
    console.error('Image download error:', err);
    result.error = err;
    return result;
  }

  try {
    const image = await tf.node.decodeImage(pic.data, 3);
    const predictions = await nsfwModel.classify(image);
    image.dispose();
    result = predictions;
  } catch (err) {
    console.log('Prediction Error:', err);
    result.error = err;
    return result;
  }

  return result;
};

const init = async () => {
  if (!AUTHORIZATION_KEY)
    console.log('No Authorization Key supplied, all requests will go through.');

  try {
    nsfwModel = await nsfw.load(NSFW_MODEL, {
      size: NSFW_MODEL_SIZE,
    });
  } catch (err) {
    console.log(err);
  }
};

init().then(() => app.listen(PORT, () => console.log(`Listening on ${DOMAIN}:${PORT}`)));
