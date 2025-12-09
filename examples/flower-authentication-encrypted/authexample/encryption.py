"""Encryption utilities for model weights in Flower federated learning."""

import io
import pickle
from typing import Dict, OrderedDict

import numpy as np
import torch
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class ModelEncryption:
    """Handle encryption and decryption of model weights."""

    def __init__(self, key: bytes):
        """Initialize encryption with a symmetric key.

        Parameters
        ----------
        key : bytes
            32-byte encryption key for AES-256-GCM
        """
        if len(key) != 32:
            raise ValueError("Encryption key must be 32 bytes for AES-256")
        self.aesgcm = AESGCM(key)

    @staticmethod
    def generate_key() -> bytes:
        """Generate a random 256-bit encryption key.

        Returns
        -------
        bytes
            32-byte encryption key
        """
        return AESGCM.generate_key(bit_length=256)

    def encrypt_state_dict(
        self, state_dict: OrderedDict[str, torch.Tensor]
    ) -> np.ndarray:
        """Encrypt a PyTorch state dictionary.

        Parameters
        ----------
        state_dict : OrderedDict[str, torch.Tensor]
            Model state dictionary to encrypt

        Returns
        -------
        np.ndarray
            Single numpy array containing nonce (12 bytes) + encrypted data
        """
        # Serialize state dict to bytes
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        plaintext = buffer.getvalue()

        # Generate a random nonce (12 bytes for GCM)
        nonce = np.random.bytes(12)

        # Encrypt the serialized data
        ciphertext = self.aesgcm.encrypt(nonce, plaintext, associated_data=None)

        # Combine nonce and ciphertext into single bytes object
        combined = nonce + ciphertext

        # Convert to numpy array for ArrayRecord compatibility
        return np.frombuffer(combined, dtype=np.uint8)

    def decrypt_state_dict(
        self, encrypted_array: np.ndarray
    ) -> OrderedDict[str, torch.Tensor]:
        """Decrypt an encrypted state dictionary.

        Parameters
        ----------
        encrypted_array : np.ndarray
            Numpy array containing nonce (first 12 bytes) + encrypted data

        Returns
        -------
        OrderedDict[str, torch.Tensor]
            Decrypted model state dictionary
        """
        # Convert numpy array back to bytes
        combined = encrypted_array.tobytes()

        # Extract nonce (first 12 bytes) and ciphertext (rest)
        nonce = combined[:12]
        ciphertext = combined[12:]

        # Decrypt the data
        plaintext = self.aesgcm.decrypt(nonce, ciphertext, associated_data=None)

        # Deserialize back to state dict
        buffer = io.BytesIO(plaintext)
        state_dict = torch.load(buffer, weights_only=True)

        return state_dict

    def encrypt_arrays(self, arrays: np.ndarray) -> Dict[str, bytes]:
        """Encrypt numpy arrays.

        Parameters
        ----------
        arrays : np.ndarray
            Arrays to encrypt

        Returns
        -------
        Dict[str, bytes]
            Dictionary with encrypted data and nonce
        """
        # Serialize arrays to bytes
        plaintext = pickle.dumps(arrays)

        # Generate a random nonce
        nonce = np.random.bytes(12)

        # Encrypt the serialized data
        ciphertext = self.aesgcm.encrypt(nonce, plaintext, associated_data=None)

        return {"ciphertext": ciphertext, "nonce": nonce}

    def decrypt_arrays(self, encrypted_data: Dict[str, bytes]) -> np.ndarray:
        """Decrypt encrypted numpy arrays.

        Parameters
        ----------
        encrypted_data : Dict[str, bytes]
            Dictionary with encrypted data and nonce

        Returns
        -------
        np.ndarray
            Decrypted arrays
        """
        ciphertext = encrypted_data["ciphertext"]
        nonce = encrypted_data["nonce"]

        # Decrypt the data
        plaintext = self.aesgcm.decrypt(nonce, ciphertext, associated_data=None)

        # Deserialize back to arrays
        arrays = pickle.loads(plaintext)

        return arrays


def load_encryption_key(key_path: str) -> bytes:
    """Load encryption key from file.

    Parameters
    ----------
    key_path : str
        Path to the encryption key file

    Returns
    -------
    bytes
        32-byte encryption key
    """
    with open(key_path, "rb") as f:
        return f.read()


def save_encryption_key(key: bytes, key_path: str) -> None:
    """Save encryption key to file.

    Parameters
    ----------
    key : bytes
        32-byte encryption key
    key_path : str
        Path to save the encryption key file
    """
    with open(key_path, "wb") as f:
        f.write(key)
