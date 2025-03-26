-- Thiết lập database Supabase

-- Bảng lưu trữ thông tin tệp tin
CREATE TABLE IF NOT EXISTS public.files (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  filename TEXT NOT NULL,
  file_path TEXT NOT NULL,
  chunking_method TEXT DEFAULT 'hybrid',
  chunk_count INTEGER DEFAULT 0,
  status TEXT DEFAULT 'active',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Index cho việc tìm kiếm tệp tin theo user_id
CREATE INDEX IF NOT EXISTS idx_files_user_id ON public.files(user_id);

-- Bảng lưu trữ cuộc hội thoại
CREATE TABLE IF NOT EXISTS public.chats (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  title TEXT NOT NULL,
  last_message TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Index cho việc tìm kiếm cuộc hội thoại theo user_id
CREATE INDEX IF NOT EXISTS idx_chats_user_id ON public.chats(user_id);

-- Bảng lưu trữ tin nhắn
CREATE TABLE IF NOT EXISTS public.messages (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  chat_id UUID NOT NULL REFERENCES public.chats(id) ON DELETE CASCADE,
  role TEXT NOT NULL,
  content TEXT NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Index cho việc tìm kiếm tin nhắn theo chat_id
CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON public.messages(chat_id);

-- Thiết lập RLS (Row Level Security) cho bảng files
CREATE POLICY "Chỉ người dùng có thể xem tệp tin của mình" ON public.files
  FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Chỉ người dùng có thể thêm tệp tin của mình" ON public.files
  FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Chỉ người dùng có thể cập nhật tệp tin của mình" ON public.files
  FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Chỉ người dùng có thể xóa tệp tin của mình" ON public.files
  FOR DELETE USING (auth.uid() = user_id);

-- Thiết lập RLS (Row Level Security) cho bảng chats
CREATE POLICY "Chỉ người dùng có thể xem cuộc hội thoại của mình" ON public.chats
  FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Chỉ người dùng có thể thêm cuộc hội thoại của mình" ON public.chats
  FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Chỉ người dùng có thể cập nhật cuộc hội thoại của mình" ON public.chats
  FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Chỉ người dùng có thể xóa cuộc hội thoại của mình" ON public.chats
  FOR DELETE USING (auth.uid() = user_id);

-- Thiết lập RLS (Row Level Security) cho bảng messages
CREATE POLICY "Chỉ người dùng có thể xem tin nhắn trong cuộc hội thoại của mình" ON public.messages
  FOR SELECT USING (
    auth.uid() IN (
      SELECT user_id FROM public.chats WHERE id = chat_id
    )
  );

CREATE POLICY "Chỉ người dùng có thể thêm tin nhắn vào cuộc hội thoại của mình" ON public.messages
  FOR INSERT WITH CHECK (
    auth.uid() IN (
      SELECT user_id FROM public.chats WHERE id = chat_id
    )
  );

CREATE POLICY "Chỉ người dùng có thể cập nhật tin nhắn trong cuộc hội thoại của mình" ON public.messages
  FOR UPDATE USING (
    auth.uid() IN (
      SELECT user_id FROM public.chats WHERE id = chat_id
    )
  );

CREATE POLICY "Chỉ người dùng có thể xóa tin nhắn trong cuộc hội thoại của mình" ON public.messages
  FOR DELETE USING (
    auth.uid() IN (
      SELECT user_id FROM public.chats WHERE id = chat_id
    )
  );

-- Bật RLS trên tất cả các bảng
ALTER TABLE public.files ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.chats ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.messages ENABLE ROW LEVEL SECURITY;

-- Trigger để tự động cập nhật updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
   NEW.updated_at = now();
   RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_files_updated_at
BEFORE UPDATE ON public.files
FOR EACH ROW EXECUTE PROCEDURE update_updated_at_column();

CREATE TRIGGER update_chats_updated_at
BEFORE UPDATE ON public.chats
FOR EACH ROW EXECUTE PROCEDURE update_updated_at_column(); 