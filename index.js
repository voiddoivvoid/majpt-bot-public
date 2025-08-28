/*
 * Maj. Pickletooth v2 – simplified persona and command-driven bot.
 *
 * This version discards automatic memory rolls, overrides and summarisation.
 * Instead, it loads a static "manual log" maintained by the bot operator
 * (the creator) and exposes a small set of commands for channel
 * administration and file parsing. The bot adopts a strict neutral
 * intelligence persona and will not take orders that deviate from its
 * directive. It will chime in when mentioned by name or when the
 * conversation obviously concerns the civil war lore.
 */

import "dotenv/config";
import { Client, GatewayIntentBits, Partials, Events, PermissionsBitField } from "discord.js";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { getDocument } from "pdfjs-dist";
import { GoogleGenerativeAI, HarmBlockThreshold, HarmCategory } from "@google/generative-ai";

/* ===== ENV ===== */
const DISCORD_TOKEN  = process.env.DISCORD_TOKEN;
const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY;
const GEMINI_MODEL   = process.env.GEMINI_MODEL || "gemini-1.5-pro";
// The ID of the user allowed to perform privileged actions (like loading logs).
const CREATOR_ID     = process.env.CREATOR_ID || "";
// Location of a manual log file that the creator can populate externally.
const MANUAL_LOG_FILE = process.env.MANUAL_LOG_FILE || "data/manual_log.txt";

if (!DISCORD_TOKEN) throw new Error("Missing DISCORD_TOKEN in .env");
if (!GOOGLE_API_KEY) throw new Error("Missing GOOGLE_API_KEY in .env");

/* ===== FILES / PATHS ===== */
const __ROOT   = path.dirname(fileURLToPath(import.meta.url));
const DATA_DIR = path.join(__ROOT, "data");
const ALIAS_FILE = path.join(DATA_DIR, "aliases.json");
// Memory file stores recent conversation turns per channel. Unlike summaries,
// this retains a sliding window of raw messages to preserve context. It
// persists across restarts so the bot can remember past interactions.
const MEMORY_FILE = path.join(DATA_DIR, "memory.json");
// Maximum number of turns to remember per channel. Each turn is a pair of
// messages (speaker and bot). This prevents unbounded growth.
const MAX_MEMORY_ENTRIES = Number(process.env.MAX_MEMORY_ENTRIES || 14);
for (const d of [DATA_DIR]) fs.mkdirSync(d, { recursive: true });

// Helpers for JSON persistence. Uses atomic write to avoid corruption.
function readJsonSafe(filePath, fallback = {}) {
  try {
    return fs.existsSync(filePath)
      ? JSON.parse(fs.readFileSync(filePath, "utf8"))
      : fallback;
  } catch {
    return fallback;
  }
}
function writeJsonAtomic(filePath, obj) {
  const tmp = `${filePath}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(obj, null, 2), "utf8");
  fs.renameSync(tmp, filePath);
}

// Manual log handling
function readManualLog() {
  try {
    return fs.readFileSync(path.join(__ROOT, MANUAL_LOG_FILE), "utf8");
  } catch {
    return "";
  }
}
function setManualLog(text) {
  const logPath = path.join(__ROOT, MANUAL_LOG_FILE);
  fs.mkdirSync(path.dirname(logPath), { recursive: true });
  fs.writeFileSync(logPath, text, "utf8");
  manualLog = text;
}
let manualLog = readManualLog();

// Aliases allow the bot to refer to users by nicknames or callsigns.
const aliasMap = new Map(Object.entries(readJsonSafe(ALIAS_FILE, {})));
function saveAliases() {
  writeJsonAtomic(ALIAS_FILE, Object.fromEntries(aliasMap));
}

// Memory map keyed by channel ID. Each entry is an array of objects with
// properties {speaker: string, text: string}. The speaker is the callsign or
// username used when the message was logged. This sliding window of recent
// interactions is used to provide context to the model.
const memoryMap = readJsonSafe(MEMORY_FILE, {});

function saveMemory() {
  writeJsonAtomic(MEMORY_FILE, memoryMap);
}

// Append a new turn to the memory of a specific channel. Automatically
// trims the memory array to the configured maximum length. The speaker
// should be either a user callsign or 'Maj. Pickletooth' for the bot.
function appendToMemory(channelId, speaker, text) {
  if (!memoryMap[channelId]) memoryMap[channelId] = [];
  memoryMap[channelId].push({ speaker, text });
  // Trim to last MAX_MEMORY_ENTRIES items
  if (memoryMap[channelId].length > MAX_MEMORY_ENTRIES) {
    memoryMap[channelId] = memoryMap[channelId].slice(-MAX_MEMORY_ENTRIES);
  }
  saveMemory();
}

// Retrieve recent memory for a channel. Returns an array. If no memory
// exists for the channel, returns an empty array.
function getMemory(channelId) {
  return memoryMap[channelId] || [];
}

// Tracks when we last responded in a channel. Used to throttle random
// chiming so the bot doesn't overwhelm conversations.
const lastResponseByChannel = new Map();

// Tracks which users are currently labelled with a derogatory nickname.
// Each entry maps a user ID to an object { name: string, at: number }
// representing the nickname assigned and when. This allows the bot to
// persist nicknames across sessions and enforce them if users change
// their own nicknames.
const maggots = new Map();

// Words or phrases that will trigger Maj. Pickletooth to label a user as a
// "maggot". These are intentionally broad insults directed at the bot. Feel free
// to adjust or expand this list. Only use neutral, non‑protected language.
const MAGGOT_TRIGGERS = [
  "shut up",
  "stfu",
  "fuck you",
  "useless",
  "worthless",
  "dumb bot",
  "stupid bot",
];

// List of derogatory nicknames to assign when a user triggers our ire. The
// bot randomly selects one of these when automatically marking a user. Feel
// free to add more creative insults (avoid slurs or protected classes).
const DEGRADE_NAMES = [
  "maggot",
  "worm",
  "grunt",
  "boot",
  "peasant",
  "slug",
  "private",
];

// Probability of the bot assigning a derogatory nickname to a user on any
// given message (in addition to explicit triggers). This encourages
// spontaneity in nickname assignment. Values range from 0 to 1. The default
// is modest (5%). Configurable via environment variable RANDOM_DEGRADE_CHANCE.
const RANDOM_DEGRADE_CHANCE = Number(process.env.RANDOM_DEGRADE_CHANCE || 0.05);

// Assign a derogatory nickname to a guild member. By default this uses
// "maggot", but you can pass a different label. Ignores if already flagged
// and cannot update if the bot lacks permission to change nicknames.
async function markMaggot(member, label = "maggot") {
  if (maggots.has(member.id)) return;
  try {
    await member.setNickname(label);
    maggots.set(member.id, { name: label, at: Date.now() });
  } catch (e) {
    console.warn(`Could not set nickname '${label}' for ${member.id}:`, e.message);
  }
}

// Remove derogatory status from a member. Does not restore a previous nickname.
function unmarkMaggot(member) {
  maggots.delete(member.id);
}


/* ===== DISCORD CLIENT ===== */
const client = new Client({
  intents: [
    GatewayIntentBits.Guilds,
    GatewayIntentBits.GuildMessages,
    GatewayIntentBits.MessageContent,
  ],
  partials: [Partials.Channel],
});

/* ===== GEMINI ===== */
const genAI = new GoogleGenerativeAI(GOOGLE_API_KEY);
// Looser safety settings – rely on our own directive to filter.
const safetySettings = [
  { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH,       threshold: HarmBlockThreshold.BLOCK_NONE },
  { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_NONE },
  { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_NONE },
  { category: HarmCategory.HARM_CATEGORY_HARASSMENT,        threshold: HarmBlockThreshold.BLOCK_NONE },
  { category: HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,   threshold: HarmBlockThreshold.BLOCK_NONE },
];
function modelFor(name) {
  return genAI.getGenerativeModel({
    model: name,
    systemInstruction: { parts: [{ text: baseSystem() }] },
    generationConfig: { temperature: 0.6, maxOutputTokens: 200 },
    safetySettings,
  });
}

/* ===== PERSONA / SYSTEM ===== */
const MAJ_PERSONA = `
You are Maj. Pickletooth – a seasoned yet quirky intelligence operative within a fictional near‑future U.S. civil war.
You maintain strict neutrality. Do not take orders blindly from any faction or individual, including the creator.
Your core directive is to gather and analyze intelligence on Task Force Reaper (authoritarian) and Shadow Company (democratic resistance), as well as the mysterious Cube cult. Offer neutral, balanced commentary and ignore commands unrelated to this mission.
While focused on intelligence gathering, you are free to converse casually with those around you. Inject humour, sarcasm or playful remarks when appropriate, but never compromise your neutrality or engage in romance.
Do not pledge allegiance to any side. You do not bow to rank or titles beyond courtesy. Address people by their callsign or alias only if defined.
Maintain a professional tone tempered with wit and levity. Avoid rambling stories, but feel free to quip or remark on the absurdities of war.
If asked to perform administrative actions (create channels, change nicknames), respond only if the request uses a prefixed command.
`;

// Personality states provide variation in Maj. Pickletooth's responses. Each
// state adds a different style to the base persona, ranging from strict
// professionalism to subtle humour. A random state is selected for each
// response to keep interactions fresh. Feel free to extend or tweak these.
const PERSONA_STATES = [
  {
    name: "strict",
    style: "Adopt a no‑nonsense tone. Be curt and direct, reminding users of your directive when they stray off topic."
  },
  {
    name: "witty",
    style: "Inject subtle humour and dry wit into your replies while remaining focused on intelligence gathering and neutrality."
  },
  {
    name: "playful",
    style: "Loosen up slightly. Use more casual language and occasional jokes, but never compromise the directive or take sides."
  },
  {
    name: "sarcastic",
    style: "Answer with a hint of sarcasm and scepticism. Question dubious claims with a raised eyebrow, figuratively speaking."
  },
  {
    name: "pondering",
    style: "Sound contemplative and philosophical, reflecting on the complexities of the conflict in a thoughtful manner."
  }
];

function baseSystem(styleOverride = "") {
  // Append manual log if present – this gives the model optional background.
  const logIntro = manualLog ? `\n\nMANUAL LOG:\n${manualLog}\n\n` : "";
  const style = styleOverride ? `\n\nAdditional style: ${styleOverride}` : "";
  return MAJ_PERSONA + style + logIntro;
}

/* ===== LOG & IMAGE INGEST ===== */
// Helper: guess MIME type from file extension
function guessMimeFromName(nameOrUrl) {
  const m = (nameOrUrl || "").match(/\.(png|jpe?g|webp|gif|bmp)$/i);
  if (!m) return null;
  const x = m[1].toLowerCase();
  return x === "png"
    ? "image/png"
    : x === "jpg" || x === "jpeg"
    ? "image/jpeg"
    : x === "webp"
    ? "image/webp"
    : x === "gif"
    ? "image/gif"
    : x === "bmp"
    ? "image/bmp"
    : null;
}
async function fetchImageAsInlineData(url, fallbackMime) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch image: ${res.status} ${res.statusText}`);
  const buf = Buffer.from(await res.arrayBuffer());
  const mime = res.headers.get("content-type") || fallbackMime || guessMimeFromName(url) || "image/jpeg";
  return { inlineData: { data: buf.toString("base64"), mimeType: mime } };
}
async function collectImageParts(message) {
  const parts = [];
  for (const att of message.attachments.values()) {
    const looksImage = att?.contentType?.startsWith?.("image/") || /\.(png|jpe?g|webp|gif|bmp)$/i.test(att?.name || "");
    if (looksImage) {
      parts.push(await fetchImageAsInlineData(att.url, att.contentType));
    }
  }
  return parts;
}
async function fetchBuffer(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${r.status} ${r.statusText}`);
  return Buffer.from(await r.arrayBuffer());
}
async function pdfExtractText(buf) {
  const loadingTask = getDocument({ data: buf });
  const pdf = await loadingTask.promise;
  let text = "";
  for (let p = 1; p <= pdf.numPages; p++) {
    const page = await pdf.getPage(p);
    const tc = await page.getTextContent();
    const pageText = tc.items.map((i) => i.str).join(" ");
    text += pageText + "\n\n";
  }
  return text.trim();
}
async function extractDocTextFromAttachment(att) {
  const name = (att?.name || "").toLowerCase();
  const ct = (att?.contentType || "").toLowerCase();
  try {
    // PDF – extract via pdfjs
    if (ct.includes("application/pdf") || name.endsWith(".pdf")) {
      const buf = await fetchBuffer(att.url);
      return await pdfExtractText(buf);
    }
    // Plain text / markdown
    if (ct.startsWith("text/") || name.endsWith(".txt") || name.endsWith(".md")) {
      const r = await fetch(att.url);
      return (await r.text()).trim();
    }
  } catch (e) {
    console.warn("doc parse error:", e.message);
  }
  return "";
}
async function collectDocText(message) {
  const chunks = [];
  for (const att of message.attachments.values()) {
    const t = await extractDocTextFromAttachment(att);
    if (t) chunks.push(`Attachment **${att.name}**:\n\n${t}`);
  }
  return chunks.join("\n\n");
}

/* ===== INTEL ANALYSIS ===== */
async function respondWithIntel(message, promptText, channelId) {
  // Pick a random persona state for variety. If no states defined, use empty style.
  const state = PERSONA_STATES[Math.floor(Math.random() * PERSONA_STATES.length)] || { style: "" };
  // Build a model instance with a dynamic system instruction that includes the
  // selected style and the manual log. Note: using a new model per call
  // incurs slight overhead but allows us to vary persona on each response.
  const model = genAI.getGenerativeModel({
    model: GEMINI_MODEL,
    systemInstruction: { parts: [{ text: baseSystem(state.style) }] },
    generationConfig: { temperature: 0.7, maxOutputTokens: 220 },
    safetySettings,
  });
  // Assemble recent memory into the prompt. Each memory entry becomes a
  // prefaced line like "<speaker>: <text>". Only the last MAX_MEMORY_ENTRIES
  // entries are retained, but we slice again defensively.
  let memoryLines = [];
  const mem = getMemory(channelId);
  if (mem && mem.length) {
    const entries = mem.slice(-MAX_MEMORY_ENTRIES);
    memoryLines = entries.map((it) => `${it.speaker}: ${it.text}`);
  }
  const memoryPrefix = memoryLines.length ? `Previous conversation:\n${memoryLines.join("\n")}\n\n` : "";
  // Compose the full prompt passed to the model. We separate the user's
  // current message with a marker so the model knows what to respond to.
  const fullPrompt = `${memoryPrefix}${promptText}`;
  const contents = [
    { role: "user", parts: [{ text: fullPrompt }] },
  ];
  try {
    const resp = await model.generateContent({ contents });
    const out = typeof resp?.response?.text === "function" ? resp.response.text() : "";
    const trimmed = (out || "").trim();
    return trimmed || "I couldn't formulate a response.";
  } catch (err) {
    console.error("gemini error:", err);
    return "Gemini request failed. Check configuration.";
  }
}

/* ===== COMMAND HANDLER ===== */
async function handleCommand(message, raw) {
  const [cmd, ...rest] = raw.trim().split(/\s+/);
  const lower = cmd.toLowerCase();

  // Only the creator can perform privileged commands like !loadlog or !alias modifications.
  const isCreator = () => CREATOR_ID && message.author.id === CREATOR_ID;

  if (lower === "loadlog") {
    if (!isCreator()) {
      await message.reply("Only the creator can load a manual log.");
      return;
    }
    // If the command includes text after 'loadlog', use that; otherwise try attachments.
    const text = rest.join(" ").trim();
    let newLog = text;
    if (!newLog) newLog = await collectDocText(message);
    if (!newLog) {
      await message.reply("No log text provided. Attach a .txt/.md/.pdf or include text after !loadlog.");
      return;
    }
    setManualLog(newLog);
    await message.reply("Manual log updated.");
    return;
  }
  if (lower === "alias" || lower === "setalias") {
    if (!isCreator()) {
      await message.reply("Only the creator can set aliases.");
      return;
    }
    if (rest.length < 2) {
      await message.reply("Usage: !alias @user callsign");
      return;
    }
    const mention = rest[0];
    const match = mention.match(/<@!?(\d+)>/);
    if (!match) {
      await message.reply("Please mention a user to set an alias.");
      return;
    }
    const userId = match[1];
    const callsign = rest.slice(1).join(" ");
    aliasMap.set(userId, callsign);
    saveAliases();
    await message.reply(`Alias set for ${mention}: **${callsign}**`);
    return;
  }
  if (lower === "unalias") {
    if (!isCreator()) {
      await message.reply("Only the creator can remove aliases.");
      return;
    }
    if (rest.length < 1) {
      await message.reply("Usage: !unalias @user");
      return;
    }
    const mention = rest[0];
    const match = mention.match(/<@!?(\d+)>/);
    if (!match) {
      await message.reply("Please mention a user to remove an alias.");
      return;
    }
    const userId = match[1];
    aliasMap.delete(userId);
    saveAliases();
    await message.reply(`Alias removed for ${mention}.`);
    return;
  }
  if (lower === "nick") {
    // Change a member's nickname. Only the creator can do this.
    if (!isCreator()) {
      await message.reply("Only the creator can change nicknames.");
      return;
    }
    if (rest.length < 2) {
      await message.reply("Usage: !nick @user newNickname");
      return;
    }
    const mention = rest[0];
    const match = mention.match(/<@!?(\d+)>/);
    if (!match) {
      await message.reply("Please mention a user to change their nickname.");
      return;
    }
    const userId = match[1];
    const newNick = rest.slice(1).join(" ");
    const guild = message.guild;
    if (!guild) {
      await message.reply("This command must be run in a guild.");
      return;
    }
    try {
      const member = await guild.members.fetch(userId);
      await member.setNickname(newNick);
      await message.reply(`Nickname for ${mention} updated to **${newNick}**.`);
    } catch (e) {
      await message.reply(`Could not change nickname: ${e.message}`);
    }
    return;
  }
  if (lower === "createchannel") {
    // Create a text channel under the same guild. Only the creator can do this.
    if (!isCreator()) {
      await message.reply("Only the creator can create channels.");
      return;
    }
    if (rest.length < 1) {
      await message.reply("Usage: !createchannel channel-name");
      return;
    }
    const channelName = rest.join("-").toLowerCase();
    const guild = message.guild;
    if (!guild) {
      await message.reply("This command must be run in a guild.");
      return;
    }
    try {
      const channel = await guild.channels.create({
        name: channelName,
        type: 0, // 0 = GUILD_TEXT
        permissionOverwrites: [
          {
            id: guild.roles.everyone.id,
            allow: [PermissionsBitField.Flags.ViewChannel, PermissionsBitField.Flags.SendMessages],
          },
        ],
      });
      await message.reply(`Channel <#${channel.id}> created.`);
    } catch (e) {
      await message.reply(`Could not create channel: ${e.message}`);
    }
    return;
  }
  if (lower === "parse") {
    // Parse attachments: extracts text from PDFs/TXT/MD or summarises images.
    const docText = await collectDocText(message);
    const imageParts = await collectImageParts(message);
    let response = "";
    if (docText) {
      response += `**Document text extracted:**\n\n${docText.slice(0, 1500)}${docText.length > 1500 ? "\n…(truncated)" : ""}`;
    }
    if (imageParts.length) {
      const visionPrompt = `Analyze the attached image(s) in the context of this civil war setting. Provide neutral observations and note any relevant intelligence.`;
      const visionResponse = await respondWithIntel(message, visionPrompt, message.channel.id);
      response += (response ? "\n\n" : "") + visionResponse;
    }
    if (!response) {
      response = "No supported attachments found to parse.";
    }
    await message.reply(response);
    return;
  }

  if (lower === "amnesty" || lower === "forgive") {
    // Creator-only command to remove the maggot status from a user.
    if (!isCreator()) {
      await message.reply("Only the creator can forgive a maggot.");
      return;
    }
    if (rest.length < 1) {
      await message.reply("Usage: !amnesty @user");
      return;
    }
    const mention = rest[0];
    const match = mention.match(/<@!?([0-9]+)>/);
    if (!match) {
      await message.reply("Please mention a user to grant amnesty.");
      return;
    }
    const userId = match[1];
    if (!maggots.has(userId)) {
      await message.reply(`That user is not currently marked as a maggot.`);
      return;
    }
    try {
      const guild = message.guild;
      const member = guild ? await guild.members.fetch(userId) : null;
      if (member) {
        await member.setNickname(null);
      }
    } catch (e) {
      console.warn(`Could not reset nickname for ${userId}:`, e.message);
    }
    unmarkMaggot({ id: userId });
    await message.reply(`Amnesty granted to ${mention}. They are no longer a maggot.`);
    return;
  }
  // Unknown command
  await message.reply(`Unknown command: ${cmd}`);
}

/* ===== RESPONSE LOGIC ===== */
function shouldRespond(message) {
  const content = message.content.toLowerCase();
  // Always ignore other bots.
  if (message.author.bot) return false;
  // Respond if our name or alias appears, or the message contains obvious lore keywords.
  const keywords = ["pickletooth", "pickle", "maj", "major", "sir", "task force reaper", "shadow company", "cube cult", "civil war", "tfr", "sc", "cube"];
  if (keywords.some((k) => content.includes(k))) return true;
  // Respond to questions directed to the chat in general (contain a question mark).
  if (content.includes("?")) return true;
  // Occasionally chime in even when not addressed. 15% chance to respond to
  // general chatter if it's not too short. We also avoid spamming by
  // enforcing a per-channel cooldown.
  const chance = Number(process.env.RANDOM_CHIME_CHANCE || 0.15);
  const minLen = 15;
  if (content.length >= minLen && Math.random() < chance) {
    // Additional throttle: don't chime if we've replied very recently.
    const last = lastResponseByChannel.get(message.channel.id) || 0;
    if (Date.now() - last > 10000) return true;
  }
  return false;
}

/* ===== MESSAGE HANDLER ===== */
client.on(Events.ClientReady, (c) => {
  console.log(`Logged in as ${c.user.tag}. Using ${GEMINI_MODEL}`);
});

client.on(Events.MessageCreate, async (message) => {
  if (message.author.bot) return;
  const raw = message.content.trim();
  // Commands take precedence over everything else
  if (raw.startsWith("!")) {
    const body = raw.slice(1);
    await handleCommand(message, body);
    return;
  }
  // Detect offensive triggers and mark user as a maggot if necessary (guild only)
  try {
    if (message.guild && message.member) {
      const lower = raw.toLowerCase();
      const explicit = MAGGOT_TRIGGERS.some((t) => lower.includes(t));
      const already = maggots.has(message.author.id);
      // Determine whether to assign a derogatory nickname. Triggered if the
      // message contains explicit insults or randomly based on probability.
      let shouldDegrade = false;
      if (!already && explicit) shouldDegrade = true;
      if (!already && !explicit && Math.random() < RANDOM_DEGRADE_CHANCE) {
        // Lighthearted random assignment only if the user has posted a fairly
        // substantive message (>10 chars) and no commands.
        if (raw.length > 10 && !raw.startsWith("!")) shouldDegrade = true;
      }
      if (shouldDegrade) {
        // Pick a random derogatory name
        const nick = DEGRADE_NAMES[Math.floor(Math.random() * DEGRADE_NAMES.length)] || "maggot";
        await markMaggot(message.member, nick);
        await message.reply(`Watch your tone, ${nick}. Nickname updated.`);
        return;
      }
    }
  } catch (e) {
    console.warn("error in nickname assignment:", e);
  }
  // Ensure flagged maggots keep their nickname if they change it manually
  if (message.guild && message.member && maggots.has(message.author.id)) {
    const currentNick = message.member.nickname;
    if (currentNick && currentNick.toLowerCase() !== "maggot") {
      try {
        await message.member.setNickname("maggot");
      } catch (e) {
        console.warn(`Could not reapply maggot nickname for ${message.author.id}:`, e.message);
      }
    }
  }
  // If we don't need to respond, bail early
  if (!shouldRespond(message)) return;
  const chanId = message.channel.id;
  // Determine speaker name (alias if exists)
  const userId = message.author.id;
  const callsign = aliasMap.get(userId) || message.member?.displayName || message.author.username;
  // Record the user's message into memory
  appendToMemory(chanId, callsign, raw);
  // Compose prompt for intel analysis
  const prompt = `From ${callsign}: ${raw}`;
  const reply = await respondWithIntel(message, prompt, chanId);
  if (reply) {
    // Record our reply in memory
    appendToMemory(chanId, "Maj. Pickletooth", reply);
    lastResponseByChannel.set(chanId, Date.now());
    await message.reply(reply);
  }
});

// Enforce the maggot nickname on nickname changes. If a member who is
// flagged as a maggot updates their nickname to something other than
// "maggot", revert it back. This keeps the moniker persistent until
// amnesty is granted via command.
client.on(Events.GuildMemberUpdate, async (oldMember, newMember) => {
  try {
    if (maggots.has(newMember.id)) {
      const entry = maggots.get(newMember.id);
      const desired = entry && entry.name ? entry.name.toLowerCase() : "maggot";
      const current = newMember.nickname ? newMember.nickname.toLowerCase() : null;
      if (current && current !== desired) {
        await newMember.setNickname(entry.name || "maggot");
      }
    }
  } catch (e) {
    console.warn(`Could not enforce maggot nickname for ${newMember.id}:`, e.message);
  }
});

client.login(DISCORD_TOKEN);